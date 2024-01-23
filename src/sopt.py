import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer
import numpy as np
import subprocess
import tempfile

import generate_c

from pynvml import *
nvmlInit()

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
GENERATE_EVERY  = 10
NUM_TOKENS = 4200 + 2
ENC_SEQ_LEN = 512
DEC_SEQ_LEN = 1024 + 1

# helpers

def cycle():
  while True:

    prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
    src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
    tgt = torch.cat((prefix, src, src), 1)
    src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().cuda()
    yield (src, tgt, src_mask)

# instantiate model

model = XTransformer(
  dim = 128*3*3*2,
  tie_token_emb = True,
  enc_attn_flash = True,
  dec_attn_flash = True,
  return_tgt_loss = True,
  enc_num_tokens=NUM_TOKENS,
  enc_depth = 2+2+2+2,
  enc_heads = 2+2+2+2,
  enc_max_seq_len = ENC_SEQ_LEN,
  dec_num_tokens = NUM_TOKENS,
  dec_depth = 2+2+2+2,
  dec_heads = 2+2+2+2,
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

  src, tgt, src_mask = next(cycle())

  loss = model(src, tgt, mask=src_mask)
  loss.backward()
  print(f'{i}: {loss.item()}')

  optim.step()
  optim.zero_grad()

  if i != 0 and i % GENERATE_EVERY == 0:
    model.eval()
    src, _, src_mask = next(cycle())
    src, src_mask = src[:1], src_mask[:1]
    start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

    sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask = src_mask)
    incorrects = (src != sample).abs().sum()

    print(f"input:  ", src)
    print(f"predicted output:  ", sample)
    print(f"incorrects: {incorrects}")


    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total//1024//1024}MB')
    print(f'free     : {info.free//1024//1024}MB')
    print(f'used     : {info.used//1024//1024}MB')
