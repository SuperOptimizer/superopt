import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer
import numpy as np
import subprocess
import tempfile
import generate_c
import gzip
import csv
import multiprocessing
from functools import wraps
import time

import sys
from types import ModuleType, FunctionType
from gc import get_referents


from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from riscv_sopt import NUM_TOKENS, tokenize_prog, tkn, constprop_gen, detokenize_prog
from create_optimization_dataset import compile

NUM_BATCHES = int(1e5)
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
GENERATE_EVERY  = 10
NUM_TOKENS = NUM_TOKENS
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
elif '4090' in torch.cuda.get_device_name():
  DIM = 1024
  BATCH_SIZE = 64
  GENERATE_EVERY = 200
  ENC_DEPTH = 8
  ENC_HEADS = 8
  DEC_DEPTH = 8
  DEP_HEADS = 8
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


def optim_db_save(c_code:str, unopt:list, opt:list):
  with open('/tmp/sopt/db.csv','a+') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writerow({'c':c_code,'unopt':unopt[:unopt.index(tkn('PAD'))],'opt':opt[:opt.index(tkn('PAD'))]})

def func(uuid):
  while True:
    prog = None
    while prog is None:
      prog = compile(uuid)
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


training_data = []
async_result = None
pool = multiprocessing.Pool()

def initiate_async_call(batchsize):
  global async_result, pool
  async_result = pool.map_async(func, list(range(batchsize * 64)))


def getbatch(batchsize, uuid):
  global training_data, async_result, pool
  # Check if training_data is empty to decide whether to wait for the async operation
  if len(training_data) == 0:
    if async_result is None:
      initiate_async_call(batchsize)
    training_data = async_result.get()
    initiate_async_call(batchsize)

  # Retrieve the required batch from training_data
  ret = training_data[:batchsize]
  # Update training_data by removing the batch that was just retrieved
  training_data = training_data[batchsize:]
  return ret


def cycle():
  uuid = 0

  while True:
    batch = getbatch(BATCH_SIZE, uuid)
    uuid += BATCH_SIZE

    mysrc = torch.tensor(list(x[0] for x in batch)).long().cuda()
    mytgt = torch.tensor(list(x[1] for x in batch)).long().cuda()
    mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().cuda()
    yield (mysrc, mysrc_mask, mytgt)

# instantiate model
@timeit
def main():
  nvmlInit()

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
  ).cuda()

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params//1024//1024}M {params//1024}K ")


  # optimizer

  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # training

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, src_mask, tgt = next(cycle())
    #print(src,tgt,src_mask)
    loss = model(src, tgt, mask=src_mask)
    loss.backward()
    print(f'{i}: {loss.item()}')

    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
      #torch.save({'epoch':i, 'model_state_dict':model.state_dict(),'optimizer_state_dict':optim.state_dict(),'loss':loss.item()}, f'/tmp/sopt/checkpoint_{i}.pt')
      model.eval()
      src, src_mask, tgt = next(cycle())
      src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]
      #start_tokens = (torch.ones((1, 1)) * 1).long().cuda()
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

      h = nvmlDeviceGetHandleByIndex(0)
      info = nvmlDeviceGetMemoryInfo(h)
      print(f'total    : {info.total//1024//1024}MB')
      print(f'free     : {info.free//1024//1024}MB')
      print(f'used     : {info.used//1024//1024}MB')

if __name__ == '__main__':
  main()