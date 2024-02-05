import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer
import numpy as np
import subprocess
import tempfile

import generate_c

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()

# constants

from riscv_sopt import NUM_TOKENS, tokenize_prog, tkn
from create_optimization_dataset import compile
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
GENERATE_EVERY  = 10
NUM_TOKENS = NUM_TOKENS
ENC_SEQ_LEN = 200
DEC_SEQ_LEN = 200

# helpers

def cycle():
  uuid = 0
  prog = None
  while True:
    batch = []
    while len(batch) < BATCH_SIZE:
      while prog is None:
        prog = compile(uuid)
      unopt_tokenized = tokenize_prog(prog['unopt'], True, 200)
      opt_tokenized   = tokenize_prog(prog['opt'],  False, 200)
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

      mysrc = torch.tensor([unopt_tokenized]).long().cuda()
      mytgt = torch.tensor([opt_tokenized]).long().cuda()
      mysrc_mask = torch.tensor([mysrc_mask]).bool().cuda()
      batch.append([mysrc,mysrc_mask,mytgt])

    mysrc = torch.cat(list(x[0] for x in batch), dim=0)
    mysrc_mask = torch.cat(list(x[1] for x in batch), dim=0)
    mytgt = torch.cat(list(x[2] for x in batch), dim=0)
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
    model.eval()
    src, src_mask, _ = next(cycle())
    src, src_mask = src[:1], src_mask[:1]
    #start_tokens = (torch.ones((1, 1)) * 1).long().cuda()
    start_tokens = torch.tensor([tkn('DECSTART')]).cuda()
    sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask = src_mask)
    incorrects = (src != sample).sum()

    print(f"input:  ", src)
    print(f"predicted output:  ", sample)
    print(f"incorrects: {incorrects}")

    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total//1024//1024}MB')
    print(f'free     : {info.free//1024//1024}MB')
    print(f'used     : {info.used//1024//1024}MB')
