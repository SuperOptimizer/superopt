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

from utils import ROOTDIR, timeit, report_cuda_size
from model import get_model
from riscv import tokenize

if torch.cuda.is_available():
  from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


@timeit
def infer(rank, world_size, device, unopt_asm):

  if device == 'cuda':
    nvmlInit()
  if world_size > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)

  model, dtype, encoder_len, decoder_len, batch_size, generate_every = get_model(device, tkn('PAD'), NUM_TOKENS, rank, world_size)

  if device == 'cuda':
    model = model.cuda(device=rank)
  elif device == 'mps':
    model = model.to('mps')
  elif device == 'cpu':
    model = model.to('cpu')

  if world_size > 1:
    model = FSDP(model, use_orig_params=True)

  if device in ['cuda']:
    model = torch.compile(model)

  if os.path.exists(f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt'):
    print(f"loading /{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt")
    checkpoint = torch.load(f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

  report_cuda_size()

  model.eval()

  src = tokenize(unopt_asm,True,encoder_len)
  mask = [True] * len(src)
  mask.extend([False] * (encoder_len - len(src)))
  src.extend([tkn('PAD')] * (encoder_len - len(src)))

  src = torch.tensor(src).long().to(device)
  mask = torch.tensor(mask).bool().to(device)
  start_tokens = torch.tensor([tkn('DECSTART')]).to(device)
  sample = model.generate(src, start_tokens, decoder_len, mask = mask)

  print_stmt = f'\nRANK: {rank} start\n'
  print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
  print_stmt += f"\ninput asm:  \n{unopt_asm} \n"
  print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
  print_stmt += f'\nRANK: {rank} end\n'
  print(print_stmt)

  if world_size > 1:
    torch.distributed.destroy_process_group()

def main():
  if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
    world_size = 1
  elif torch.cuda.is_available():
    device = 'cuda'
    world_size = torch.cuda.device_count()
  else:
    device = 'cpu'
    world_size = 1
  if world_size <= 1:
    infer(0, 1, device)
