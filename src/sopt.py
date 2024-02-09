import tqdm
import torch
from x_transformers import XTransformer
import numpy as np
import csv
import multiprocessing
from functools import wraps
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import gzip
import ast
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from riscv_sopt import NUM_TOKENS, tokenize_prog, tkn, constprop_gen, detokenize_prog
from create_optimization_dataset import compile

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

USERDIR = os.path.expanduser('~')

NUM_BATCHES = int(1e5)
LEARNING_RATE = 1e-4
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 128

if '2060' in torch.cuda.get_device_name():
  DIM = 512
  BATCH_SIZE = 8
  GENERATE_EVERY = 10
  ENC_DEPTH = 4
  ENC_HEADS = 4
  DEC_DEPTH = 4
  DEP_HEADS = 4
  DTYPE=torch.float16
elif 'V100' in torch.cuda.get_device_name():
  DIM = 1024
  BATCH_SIZE = 32
  GENERATE_EVERY = 100
  ENC_DEPTH = 8
  ENC_HEADS = 8
  DEC_DEPTH = 8
  DEP_HEADS = 8
  DTYPE=torch.float16
elif ('4090' in torch.cuda.get_device_name() or
      'A5000' in torch.cuda.get_device_name() or
      '3090' in torch.cuda.get_device_name()):
  DIM = 1024
  BATCH_SIZE = 64
  GENERATE_EVERY = 200
  ENC_DEPTH = 8
  ENC_HEADS = 8
  DEC_DEPTH = 8
  DEP_HEADS = 8
  DTYPE=torch.bfloat16
else:
  assert False


def getsize(obj):
  BLACKLIST = type, ModuleType, FunctionType
  """sum size of object & members."""
  if isinstance(obj, BLACKLIST):
      raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
  seen_ids = set()
  size = 0
  objects = [obj]
  while objects:
    need_referents = []
    for obj in objects:
      if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
        seen_ids.add(id(obj))
        size += sys.getsizeof(obj)
        need_referents.append(obj)
    objects = get_referents(*need_referents)
  return size

def timeit(func):
  @wraps(func)
  def timeit_wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
    return result
  return timeit_wrapper



def gen_training_entry(uuid):
  while True:
    prog = None
    while prog is None:
      prog = compile((uuid,8,30))
      #prog = constprop_gen()
    unopt_tokenized = tokenize_prog(prog['unopt'], True, ENC_SEQ_LEN)
    if unopt_tokenized is None:
      prog = None
      continue
    opt_tokenized   = tokenize_prog(prog['opt'],  False, DEC_SEQ_LEN)
    if opt_tokenized is None:
      prog = None
      continue

    mysrc_mask = []
    for x in unopt_tokenized:
      if x != tkn('PAD'):
        mysrc_mask.append(True)
      else:
        mysrc_mask.append(False)

    mytgt_mask = []
    for x in opt_tokenized:
      if x != tkn('PAD'):
        mytgt_mask.append(True)
      else:
        mytgt_mask.append(False)

    return unopt_tokenized,opt_tokenized,mysrc_mask

def cycle(training_data, db_idx):
  if len(training_data) < BATCH_SIZE:
    print("db_idx", db_idx)
    with gzip.open(f'/{ROOTDIR}/data/db_{db_idx}.csv.gz','rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = ast.literal_eval(entry['unopt'])
        opt = ast.literal_eval(entry['opt'])
        mask = [True]*len(unopt)
        mask.extend([False]*(ENC_SEQ_LEN-len(unopt)))
        unopt.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt)))
        opt.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt)))
        training_data.append((unopt,opt,mask))
      db_idx += 1
      if not os.path.exists(f'/{ROOTDIR}/data/db_{db_idx}.csv.gz'):
        db_idx = 0
  batch = training_data[:BATCH_SIZE]
  training_data = training_data[BATCH_SIZE:]

  mysrc = torch.tensor(list(x[0] for x in batch)).long().cuda()
  mytgt = torch.tensor(list(x[1] for x in batch)).long().cuda()
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().cuda()
  return mysrc, mysrc_mask, mytgt, training_data, db_idx


@timeit
def train(rank, world_size):

  nvmlInit()
  if world_size > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)

  model = XTransformer(
    dim = DIM,
    tie_token_emb = True,
    enc_attn_flash = True,
    dec_attn_flash = True,
    return_tgt_loss = True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth = ENC_DEPTH,
    enc_heads = ENC_HEADS,
    enc_max_seq_len = ENC_SEQ_LEN,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = ENC_DEPTH,
    dec_heads = ENC_HEADS,
    dec_max_seq_len = DEC_SEQ_LEN
  ).cuda(device=rank)
  if world_size > 1:
    model = FSDP(model)

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params//1024//1024}M {params//1024}K ")

  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)

  training_data = []
  db_idx = rank

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt, training_data, db_idx = cycle(training_data, db_idx)
    with torch.cuda.amp.autocast(dtype=DTYPE):
      loss = model(src, tgt, mask=src_mask)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0:
      h = nvmlDeviceGetHandleByIndex(0)
      info = nvmlDeviceGetMemoryInfo(h)
      print(f'total    : {info.total // 1024 // 1024}MB')
      print(f'free     : {info.free // 1024 // 1024}MB')
      print(f'used     : {info.used // 1024 // 1024}MB')


    if i != 0 and i % GENERATE_EVERY == 0:
      with FSDP.summon_full_params(model, writeback=False, recurse=False):
        torch.save({'epoch':i,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':loss.item(),
                    'scaler':scaler.state_dict(),
                    'scheduler':scheduler.state_dict()},
                   f'/{ROOTDIR}/checkpoint.pt')
        #TODO: this code isn't working on FSDP yet
        model.eval()
        src, src_mask, tgt, training_data, db_idx = cycle(training_data, db_idx)
        src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([tkn('DECSTART')]).cuda()
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)
        #the target output always includes the 'DECSTART' token whereas the sampled output does not
        #so shift the output left one token to delete it
        for x in range(DEC_SEQ_LEN-1):
          tgt[0,x] = tgt[0,x+1]
        incorrects = (tgt != sample).sum()

        print(f"input:  ", detokenize_prog(src.tolist()[0]))
        print(f"predicted tokens:  ", sample.tolist())
        print(f"actual tokens:     ", tgt.tolist()[0])
        print(f"predicted asm:  ", detokenize_prog(sample.tolist()))
        print(f"actual asm:     ", detokenize_prog(tgt.tolist()[0]))
        print(f"incorrects: {incorrects}")

  if world_size > 1:
    torch.distributed.destroy_process_group()

def main():
  world_size = torch.cuda.device_count()
  if world_size == 0:
    train(0,1)
  else:
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size,join=True)

if __name__ == '__main__':
  main()