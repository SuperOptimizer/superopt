from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import sys
import ast
import gzip
import torch
import random
import tqdm
import sentencepiece as spm
import torchao
from torchao.optim import _AdamW



from impl import (
  get_model, tokenize_bytes, detokenize_bytes, tokenize_hexstr, detokenize_hexstr, tkn, MODEL_SIZE,
  DTYPE, DEVICE, GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES, TMP)
from util import report_cuda_size, timeit, report_model_size, chunkify
from codegen import gen_yarpgen

CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

def save_checkpoint(model,  optim, loss):
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss.item(),},
    CHECKPOINT)

def load_checkpoint(model, optim, loss):
  if os.path.exists(CHECKPOINT):
    print(f"loading {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
  return model, optim,  loss



#todo: batch_size > 1
def yarpgen_and_cycle(sp_encoder, sp_decoder):
  done = False
  while not done:
    unopt,opt = list(gen_yarpgen(0,1))[0]
    unopt_tokens = tokenize_bytes(sp_encoder, unopt)
    opt_tokens = tokenize_bytes(sp_decoder, opt)
    if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
      print("skipping...")
      #try again
      pass
    else:
      done = True
  opt_tokens.insert(0, tkn('DECSTART'))
  mask = [True] * len(unopt_tokens)
  mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
  unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
  opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))

  batch = [[unopt_tokens, opt_tokens, mask]]
  mysrc = torch.tensor(list(x[0] for x in batch)).long().to(DEVICE)
  mytgt = torch.tensor(list(x[1] for x in batch)).long().to(DEVICE)
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(DEVICE)
  return mysrc, mysrc_mask, mytgt

def cycle(encoder_gzip, decoder_gzip, sp_encoder, sp_decoder):
  """Generator that yields batches of data directly from gzip files."""
  with gzip.open(encoder_gzip, 'rt') as f, gzip.open(decoder_gzip, 'rt') as g:
    for enc_line, dec_line in zip(f, g):

      unopt_tokens = tokenize_hexstr(sp_encoder, enc_line)
      opt_tokens = tokenize_hexstr(sp_decoder, dec_line)

      # Skip if sequences are too long
      if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
        print("skipping... unopt len {} opt len {} orig unopt len {} orig opt len {}".format(len(unopt_tokens), len(opt_tokens), len(enc_line), len(dec_line)))
        continue
      # Prepare the tokens
      opt_tokens.insert(0, tkn('DECSTART'))
      mask = [True] * len(unopt_tokens)
      mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
      unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
      opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))

      mysrc = torch.tensor([unopt_tokens]).long().to(DEVICE)
      mytgt = torch.tensor([opt_tokens]).long().to(DEVICE)
      mysrc_mask = torch.tensor([mask]).bool().to(DEVICE)
      yield mysrc, mysrc_mask, mytgt



@timeit
def train():
  sp_encoder = spm.SentencePieceProcessor(model_file=f'{ROOTDIR}/misc/encoder.model')
  sp_decoder = spm.SentencePieceProcessor(model_file=f'{ROOTDIR}/misc/decoder.model')
  encoder_gzip = f"{TMP}/encoder.txt.gzip"
  decoder_gzip = f"{TMP}/decoder.txt.gzip"
  model = get_model(tkn('PAD'))
  report_model_size(model)
  optim = torchao.optim.AdamW4bit(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
  #optim = torchao.optim._AdamW(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)
  model, optim, loss = load_checkpoint(model, optim, 0)
  data_generator = cycle(encoder_gzip, decoder_gzip, sp_encoder, sp_decoder)
  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    optim.zero_grad()
    asdf = next(data_generator)
    src, src_mask, tgt = asdf
    loss = model(src, tgt, mask=src_mask)
    loss.backward()
    optim.step()

    scheduler.step(i/NUM_BATCHES)
    print(f'{i}: {loss.item()}')

    if i == 0:
      report_cuda_size()
    if i % GENERATE_EVERY == 0:
        if i > 0:
          save_checkpoint(model,  optim, loss)
        model.eval()
        src, src_mask, tgt  = next(data_generator)
        src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([tkn('DECSTART')]).to(DEVICE)
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)
        print_stmt = ""
        print_stmt += f"\ninput tokenized:  \n{detokenize_bytes(sp_encoder, src.tolist()[0])} \n"
        print_stmt += f"\npredicted detokenized:  \n{detokenize_bytes(sp_decoder, sample.tolist())}\n"
        print_stmt += f"\nactual detokenized:     \n{detokenize_bytes(sp_decoder, tgt.tolist()[0])}\n"
        print(print_stmt)


def main():
    train()

if __name__ == '__main__':
  main()
