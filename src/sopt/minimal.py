import multiprocessing
from x_transformers import XTransformer
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from  subprocess import PIPE, Popen, run
import os
import sentencepiece as spm
import random
import sys
import string
import platform
import ast
import csv
import gzip
import base64
import shutil
import time
from functools import wraps
import torch
import random
import tqdm

from impl import (
  tokenize_char as tokenize, detokenize_char as detokenize, tkn_char as tkn, save_checkpoint, load_checkpoint,
  generate_database, DTYPE, DEVICE, GENERATE_EVERY, MODEL_SIZE, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, NUM_TOKENS,
  LEARNING_RATE, NUM_BATCHES, BATCH_SIZE, WORLD_SIZE)
from util import randstring, report_cuda_size, timeit


def cycle():
  training_data = []
  csvs = sorted(os.listdir(f'/{ROOTDIR}/cleandata/'))
  if WORLD_SIZE > 1:
    random.shuffle(csvs)
  for gz in csvs:
    with gzip.open(gz,'rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = ast.literal_eval(entry['unopt'])
        opt = ast.literal_eval(entry['opt'])
        unopt_tokens = tokenize(unopt)
        opt_tokens = tokenize(opt)
        if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
          continue
        opt_tokens.insert(0, tkn('DECSTART'))
        mask = [True] * len(unopt_tokens)
        mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
        unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
        opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))
        training_data.append([unopt_tokens, opt_tokens, mask])
    while len(training_data) > BATCH_SIZE:
      batch = training_data[:BATCH_SIZE]
      training_data = training_data[BATCH_SIZE:]
      mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
      mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
      mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
      yield mysrc, mysrc_mask, mytgt


def get_model(rank):
  size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[MODEL_SIZE]

  model = XTransformer(
    dim=[256,512,768,1024][size],
    pad_value=tkn('PAD'),
    tie_token_emb=True,
    enc_attn_flash=True,
    dec_attn_flash=True,
    return_tgt_loss=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=[4,8,12,16][size],
    enc_heads=[4,8,12,16][size],
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=[4,8,12,16][size],
    dec_heads=[4,8,12,16][size],
    dec_max_seq_len=DEC_SEQ_LEN,

    enc_attn_num_mem_kv=16,
    enc_num_memory_tokens=20,
    enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    enc_ff_swish=True,
    enc_ff_glu=True,
    enc_attn_kv_heads=2,
    enc_attn_gate_values=True,
    enc_sandwich_coef=[2,4,6,8][size],
    enc_shift_tokens=1,
    enc_use_abs_pos_emb=False,
    enc_attn_on_attn=True,
    enc_macaron=True,
    enc_resi_dual=True,
    enc_resi_dual_scale=0.1,

    dec_attn_num_mem_kv=16,
    dec_num_memory_tokens=20,
    dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    dec_ff_swish=True,
    dec_ff_glu=True,
    dec_attn_kv_heads=2,
    dec_attn_gate_values=True,
    dec_sandwich_coef=[2,4,6,8][size],
    dec_shift_tokens=1,
    dec_use_abs_pos_emb=False,
    dec_attn_on_attn=True,
    dec_macaron=True,
    dec_resi_dual=True,
    dec_resi_dual_scale=0.1,
  )

  if DEVICE == 'cuda':
    model = model.cuda(device=rank)
    if WORLD_SIZE > 1:
      model = FSDP(model, use_orig_params=True)
    model = torch.compile(model)
  else:
    model = model.to(DEVICE)

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params // 1024 // 1024}M {params // 1024}K ")

  return model


@timeit
def train(rank):

  if WORLD_SIZE > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)

  model = get_model(rank)
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)

  model, optim, loss = load_checkpoint(model, optim, 0)

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt = next(cycle())
    if DEVICE == 'cuda':
      with torch.cuda.amp.autocast(dtype=DTYPE):
        loss = model(src, tgt, mask=src_mask)
      scaler.scale(loss).backward()
      scaler.step(optim)
      scaler.update()
    else:
      loss = model(src, tgt, mask=src_mask)
      loss.backward()
      optim.step()
    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0 and DEVICE == 'cuda':
      report_cuda_size()
    if i % GENERATE_EVERY == 0:
      save_checkpoint(model,  optim, loss, scaler, scheduler)
      model.eval()
      src, src_mask, tgt = next(cycle())
      src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
      start_tokens = torch.tensor([tkn('DECSTART')]).to(DEVICE)
      sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)

      print_stmt = f'\nRANK: {rank} start\n'
      print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
      print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
      print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0])}\n"
      print_stmt += f'\nRANK: {rank} end\n'
      print(print_stmt)

  if WORLD_SIZE > 1:
    torch.distributed.destroy_process_group()

def main():
  print(f'spawning {WORLD_SIZE} GPU(s)')
  if WORLD_SIZE == 1:
    train(0)
  else:
    torch.multiprocessing.spawn(train, args=(), nprocs=WORLD_SIZE,join=True)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("you must specify a trask: train, infer, gen")
    print("defaulting to train")
    sys.argv.append("train")
  if sys.argv[1] == 'train':
    main()
  elif sys.argv[1] == 'gen':
    generate_database()
  elif sys.argv[1] == 'infer':
    pass
