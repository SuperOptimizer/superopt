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

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()

# constants

from riscv_sopt import NUM_TOKENS, tokenize_prog, tkn, constprop_gen, detokenize_prog
from create_optimization_dataset import compile
NUM_BATCHES = int(1e5)
BATCH_SIZE = 8
LEARNING_RATE = 9e-4
GENERATE_EVERY  = 2
NUM_TOKENS = NUM_TOKENS
ENC_SEQ_LEN = 64
DEC_SEQ_LEN = 16

# helpers


def optim_db_save(c_code:str, unopt:list, opt:list):
  with open('/tmp/sopt/db.csv','a+') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writerow({'c':c_code,'unopt':unopt[:unopt.index(tkn('PAD'))],'opt':opt[:opt.index(tkn('PAD'))]})

def func(uuid):
  while True:
    prog = None
    while prog is None:
      #prog = compile(uuid)
      prog = constprop_gen()
    unopt_tokenized = tokenize_prog(prog['unopt'], True, 64)
    if unopt_tokenized is None:
      prog = None
      continue
    opt_tokenized   = tokenize_prog(prog['opt'],  False, 16)
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

    mysrc = torch.tensor([unopt_tokenized]).long()
    mytgt = torch.tensor([opt_tokenized]).long()
    mysrc_mask = torch.tensor([mysrc_mask]).bool()

    return prog,unopt_tokenized,opt_tokenized,mysrc,mysrc_mask,mytgt

def getbatch(batchsize, uuid):
  with multiprocessing.Pool(batchsize) as p:
    ret = p.map(func, list(range(uuid,uuid+batchsize)))
  return ret


def cycle():
  uuid = 0
  prog = None

  while True:
    batch = getbatch(BATCH_SIZE, uuid)
    uuid += BATCH_SIZE

    mysrc = torch.cat(list(x[3] for x in batch), dim=0).cuda()
    mysrc_mask = torch.cat(list(x[4] for x in batch), dim=0).cuda()
    mytgt = torch.cat(list(x[5] for x in batch), dim=0).cuda()
    yield (mysrc, mysrc_mask, mytgt)

# instantiate model

model = XTransformer(
  dim = 256,
  tie_token_emb = True,
  enc_attn_flash = True,
  dec_attn_flash = True,
  return_tgt_loss = True,
  enc_num_tokens=NUM_TOKENS,
  enc_depth = 4,
  enc_heads = 4,
  enc_max_seq_len = ENC_SEQ_LEN,
  dec_num_tokens = NUM_TOKENS,
  dec_depth = 4,
  dec_heads = 4,
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
