from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import sys
import ast
import csv
import gzip
import torch
import random
import tqdm


from impl import (
  save_checkpoint, load_checkpoint, compile_yarpgen, generate_yarpgen, clean_yarpgen, sentencepiece_train, zstd_train, gen_yarpgen, get_model,
  DTYPE, DEVICE, GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES, WORLD_SIZE, TOKENIZER)
if TOKENIZER == 'char':
  from impl import tokenize_char as tokenize, detokenize_char as detokenize, tkn_char as tkn
elif TOKENIZER == "sentencepiece":
  from impl import tokenize_sp as tokenize, detokenize_sp as detokenize, tkn_sp as tkn

from util import report_cuda_size, timeit, report_model_size, chunkify


def cycle(batch_size, training_data, txts):
  if len(training_data) < batch_size:
    txtgz = random.choice(txts)
    txts.remove(txtgz)
    if len(txts) == 0:
      txts = os.listdir(f'/{ROOTDIR}/cleandata/')
    print(f"loading /{ROOTDIR}/cleandata/{txtgz}")
    with gzip.open(f'/{ROOTDIR}/cleandata/{txtgz}','rt') as f:
      asdf = f.readlines()
      for entry in chunkify(asdf,2):
        unopt = ast.literal_eval(entry[0])
        opt = ast.literal_eval(entry[1])
        unopt_tokens = tokenize(unopt)
        opt_tokens = tokenize(opt)
        #print(f"len unopt tokens {len(unopt_tokens)} len opt tokens {len(opt_tokens)} len unopt {len(unopt)} len opt {len(opt)}")
        if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
          continue
        opt_tokens.insert(0, tkn('DECSTART'))
        mask = [True] * len(unopt_tokens)
        mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
        unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
        opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))
        training_data.append([unopt_tokens, opt_tokens, mask])
    print(f"done loading /{ROOTDIR}/cleandata/{txtgz}")
  while len(training_data) > batch_size:
    batch = training_data[:batch_size]
    training_data = training_data[batch_size:]
    mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
    mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
    mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
    return mysrc, mysrc_mask, mytgt, training_data, txts


@timeit
def train(rank):

  if WORLD_SIZE > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank,world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)

  model, batch_size = get_model(rank, tkn('PAD'))
  report_model_size(model)
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  if DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)
  model, optim, loss = load_checkpoint(model, optim, 0)

  training_data = []
  txts = os.listdir(f'/{ROOTDIR}/cleandata/')

  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()

    src, src_mask, tgt, training_data, txts = cycle(batch_size, training_data, txts)
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
      with FSDP.summon_full_params(model, writeback=False, recurse=False):
        if i > 0:
          save_checkpoint(model,  optim, loss, scaler, scheduler)
        model.eval()
        src, src_mask, tgt, training_data, txts = cycle(batch_size, training_data, txts)
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
  print(f'spawning {WORLD_SIZE} processes(s)')
  if WORLD_SIZE == 1:
    train(0)
  else:
    torch.multiprocessing.spawn(train, args=(), nprocs=WORLD_SIZE,join=True)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("you must specify a trask: train, infer, gen")
    print("defaulting to train")
    sys.argv.append("gen")
  #sentencepiece_train()
  #zstd_train()
  #if sys.argv[1] == 'train':
  main()
  #elif sys.argv[1] == 'gen':
  #generate_yarpgen()
  #compile_yarpgen()
  #clean_yarpgen()
  #elif sys.argv[1] == 'infer':
  #  pass
