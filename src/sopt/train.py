import tqdm
import torch
from x_transformers import XTransformer
import numpy as np
import csv
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import gzip
import ast
from riscv import NUM_TOKENS, tokenize, tkn, detokenize
from create_optimization_dataset import compile

from utils import ROOTDIR, timeit

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  DEVICE = 'mps'
  WORLD_SIZE = 1
elif torch.cuda.is_available():
  from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
  DEVICE = 'cuda'
  WORLD_SIZE = torch.cuda.device_count()
else:
  DEVICE = 'cpu'
  WORLD_SIZE = 1

NUM_BATCHES = int(1e5)
LEARNING_RATE = 1e-4
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 128

if DEVICE == 'cuda':
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
  elif 'A100' in torch.cuda.get_device_name():
    DIM = 2048
    BATCH_SIZE = 64
    GENERATE_EVERY = 200
    ENC_DEPTH = 20
    ENC_HEADS = 20
    DEC_DEPTH = 20
    DEP_HEADS = 20
    DTYPE=torch.bfloat16
  else:
    assert False
else:
  DIM = 512
  BATCH_SIZE = 16
  GENERATE_EVERY = 100
  ENC_DEPTH = 4
  ENC_HEADS = 4
  DEC_DEPTH = 4
  DEP_HEADS = 4
  DTYPE = torch.float16




def gen_training_entry(uuid):
  while True:
    prog = None
    while prog is None:
      prog = compile((uuid,8,30))
      #prog = constprop_gen()
    unopt_tokenized = tokenize(prog['unopt'], True, ENC_SEQ_LEN)
    if unopt_tokenized is None:
      prog = None
      continue
    opt_tokenized   = tokenize(prog['opt'],  False, DEC_SEQ_LEN)
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
    with gzip.open(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz','rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = ast.literal_eval(entry['unopt'])
        opt = ast.literal_eval(entry['opt'])
        mask = [True]*len(unopt)
        mask.extend([False]*(ENC_SEQ_LEN-len(unopt)))
        unopt.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt)))
        opt.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt)))
        training_data.append([unopt, opt, mask])
      db_idx += 1
      if not os.path.exists(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz'):
        db_idx = 0
  batch = training_data[:BATCH_SIZE]
  training_data = training_data[BATCH_SIZE:]
  mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
  mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
  return mysrc, mysrc_mask, mytgt, training_data, db_idx


@timeit
def train(rank, world_size):

  if DEVICE == 'cuda':
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
    dec_max_seq_len = DEC_SEQ_LEN)

  if DEVICE == 'cuda':
    model = model.cuda(device=rank)
  elif DEVICE == 'mps':
    model = model.to('mps')
  elif DEVICE == 'cpu':
    model = model.to('cpu')

  if world_size > 1:
    model = FSDP(model, use_orig_params=True)

  #if DEVICE in ['cuda','cpu']:
  #  model = torch.compile(model)

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params//1024//1024}M {params//1024}K ")

  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)

  training_data = []
  db_idx = rank

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt, training_data, db_idx = cycle(training_data, db_idx)
    if DEVICE == 'cuda':
      with torch.cuda.amp.autocast(dtype=DTYPE):
        loss = model(src, tgt, mask=src_mask)
      scaler.scale(loss).backward()
      scaler.step(optim)
      scaler.update()
    elif DEVICE in ['mps','cpu']:
      loss = model(src, tgt, mask=src_mask)
      loss.backward()
      optim.step()
    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0 and DEVICE == 'cuda':
      h = nvmlDeviceGetHandleByIndex(0)
      info = nvmlDeviceGetMemoryInfo(h)
      print(f'total    : {info.total // 1024 // 1024}MB')
      print(f'free     : {info.free // 1024 // 1024}MB')
      print(f'used     : {info.used // 1024 // 1024}MB')
    elif i % GENERATE_EVERY == 0:
      with FSDP.summon_full_params(model, writeback=False, recurse=False):
        if DEVICE == 'cuda':
          torch.save({'epoch':i,
                      'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optim.state_dict(),
                      'loss':loss.item(),
                      'scaler':scaler.state_dict(),
                      'scheduler':scheduler.state_dict()},
                     f'/{ROOTDIR}/checkpoint.pt')

        model.eval()
        src, src_mask, tgt, training_data, db_idx = cycle(training_data, db_idx)
        src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([tkn('DECSTART')]).to(DEVICE)
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)
        #the target output always includes the 'DECSTART' token whereas the sampled output does not
        #so shift the output left one token to delete it
        for x in range(DEC_SEQ_LEN-1):
          tgt[0,x] = tgt[0,x+1]
        incorrects = (tgt != sample).sum()
        print_stmt = f'RANK: {rank} start\n'
        print_stmt += f"input:  \n{detokenize(src.tolist()[0])} \n"
        #print_stmt += f"predicted tokens:  \n{sample.tolist()} \n"
        #print_stmt += f"actual tokens:     \n{tgt.tolist()[0]} \n"
        print_stmt += f"predicted asm:  \n{detokenize(sample.tolist())}\n"
        print_stmt += f"actual asm:     \n{detokenize(tgt.tolist()[0])}\n"
        print_stmt += f"incorrects: {incorrects}\n"
        print_stmt += f'RANK: {rank} end\n'
        print(print_stmt)

  if world_size > 1:
    torch.distributed.destroy_process_group()

def main():
  if WORLD_SIZE <= 1:
    train(0,1)
  else:
    torch.multiprocessing.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE,join=True)

if __name__ == '__main__':
  main()