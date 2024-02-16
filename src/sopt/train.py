import tqdm
import torch
import numpy as np
import csv
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import gzip
import ast
from riscv import NUM_TOKENS, tokenize, tkn, detokenize

from utils import ROOTDIR, timeit, report_cuda_size
from model import get_model

GENERATE_EVERY = 1000
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)



def cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len):
  if len(training_data) < batch_size:
    print("db_idx", db_idx)
    with gzip.open(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz','rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = ast.literal_eval(entry['unopt'])
        opt = ast.literal_eval(entry['opt'])
        unopt_asm = entry['unopt_asm']
        opt_asm = entry['opt_asm']
        mask = [True]*len(unopt)
        mask.extend([False]*(encoder_len-len(unopt)))
        unopt.extend([tkn('PAD')] * (encoder_len - len(unopt)))
        opt.extend([tkn('PAD')] * (decoder_len - len(opt)))
        training_data.append([unopt, opt, mask,unopt_asm,opt_asm])
      db_idx += 1
      if not os.path.exists(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz'):
        db_idx = 0
  batch = training_data[:batch_size]
  training_data = training_data[batch_size:]
  mysrc = torch.tensor(list(x[0] for x in batch)).long().to(device)
  mytgt = torch.tensor(list(x[1] for x in batch)).long().to(device)
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(device)
  unopt_asm = list(x[3] for x in batch)
  opt_asm = list(x[4] for x in batch)
  return mysrc, mysrc_mask, mytgt, unopt_asm, opt_asm, training_data, db_idx


@timeit
def train(rank, world_size, device):

  if world_size > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)

  model, dtype, encoder_len, decoder_len, batch_size, generate_every = get_model(device, tkn('PAD'), NUM_TOKENS,rank,world_size)

  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if device == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)

  training_data = []
  db_idx = rank

  iterations = 0
  if device == 'cuda' and os.path.exists(f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt'):
      print(f"loading /{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt")
      checkpoint = torch.load(f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt')
      model.load_state_dict(checkpoint['model_state_dict'])
      optim.load_state_dict(checkpoint['optimizer_state_dict'])
      iterations = checkpoint['iterations']
      loss = checkpoint['loss']
      db_idx = checkpoint['db_idx']

  for i in tqdm.tqdm(range(iterations,NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt, unopt_asm, opt_asm, training_data, db_idx = cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len)
    if device == 'cuda':
      with torch.cuda.amp.autocast(dtype=dtype):
        loss = model(src, tgt, mask=src_mask)
      scaler.scale(loss).backward()
      scaler.step(optim)
      scaler.update()
    elif device in ['mps','cpu']:
      loss = model(src, tgt, mask=src_mask)
      loss.backward()
      optim.step()
    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0 and device == 'cuda':
      report_cuda_size()
    if i % GENERATE_EVERY == 0:
      with FSDP.summon_full_params(model, writeback=False, recurse=False):
        if device == 'cuda':
          torch.save({
            'iterations':i,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss.item(),
            'scaler':scaler.state_dict(),
            'scheduler':scheduler.state_dict(),
            'db_idx':db_idx},
            f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}.pt')
    if i % GENERATE_EVERY == 0:
      model.eval()
      src, src_mask, tgt, unopt_asm, opt_asm, training_data, db_idx =  cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len)
      src, src_mask, tgt, unopt_asm, opt_asm = src[:1], src_mask[:1], tgt[:1], unopt_asm[:1][0] ,opt_asm[:1][0]
      start_tokens = torch.tensor([tkn('DECSTART')]).to(device)
      sample = model.generate(src, start_tokens, decoder_len, mask = src_mask)
      #the target output always includes the 'DECSTART' token whereas the sampled output does not
      #so shift the output left one token to delete it
      for x in range(decoder_len-1):
        tgt[0,x] = tgt[0,x+1]
      incorrects = (tgt != sample).sum()
      print_stmt = f'\nRANK: {rank} start\n'
      print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0])} \n"
      print_stmt += f"\ninput asm:  \n{unopt_asm} \n"
      print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist())}\n"
      print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0])}\n"
      print_stmt += f"\nactual asm:     \n{opt_asm}\n"
      print_stmt += f"\nincorrects: {incorrects}\n"
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
    train(0,1,device)
  else:
    torch.multiprocessing.spawn(train, args=(world_size,device), nprocs=world_size,join=True)

if __name__ == '__main__':
  main()