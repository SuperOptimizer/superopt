import os
import gzip
import torch
import tqdm
import sentencepiece as spm
import torchao
from torchao.optim import _AdamW


from impl import (
  get_model, tokenize_bytes, detokenize_bytes, tokenize_hexstr, detokenize_hexstr, tkn, MODEL_SIZE,
   GENERATE_EVERY, ROOTDIR, ENC_SEQ_LEN, DEC_SEQ_LEN, LEARNING_RATE, NUM_BATCHES, TMP, CHECKPOINT_EVERY, GRADIENT_ACCUMULATE_EVERY, HOMEDIR)
from util import report_cuda_size, timeit, report_model_size, chunkify
from codegen import gen_yarpgen

CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

def save_checkpoint(model,  optim, loss):
  print("saving",CHECKPOINT)
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss.item(),},
    CHECKPOINT)

def load_checkpoint(model, optim, loss=0):
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
    if len(unopt_tokens) >= ENC_SEQ_LEN-2 or len(opt_tokens) >= DEC_SEQ_LEN-2:
      print("skipping...")
      #try again
      pass
    else:
      done = True
  opt_tokens.insert(0, tkn('DECSTART'))
  opt_tokens.append(tkn('EOS'))
  mask = [True] * len(unopt_tokens)
  mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
  unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
  opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))

  batch = [[unopt_tokens, opt_tokens, mask]]
  mysrc = torch.tensor(list(x[0] for x in batch)).long().to('cuda')
  mytgt = torch.tensor(list(x[1] for x in batch)).long().to('cuda')
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to('cuda')
  return mysrc, mysrc_mask, mytgt

def cycle(sp_encoder, sp_decoder):
  """Generator that yields batches of data directly from gzip files."""
  while True:
    num_gzips = len(os.listdir(f"{HOMEDIR}/superopt_data/"))
    assert num_gzips % 2 == 0
    for i in range(num_gzips // 2):
      encoder_gzip = f"{HOMEDIR}/superopt_data/encoder_corpus_{i}.txt.gzip"
      decoder_gzip = f"{HOMEDIR}/superopt_data/decoder_corpus_{i}.txt.gzip"
      print("loading ",encoder_gzip, decoder_gzip)
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
          opt_tokens.append(tkn('EOS'))
          mask = [True] * len(unopt_tokens)
          mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
          unopt_tokens.extend([tkn('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
          opt_tokens.extend([tkn('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))

          mysrc = torch.tensor([unopt_tokens]).long().to('cuda')
          mytgt = torch.tensor([opt_tokens]).long().to('cuda')
          mysrc_mask = torch.tensor([mask]).bool().to('cuda')
          yield mysrc, mysrc_mask, mytgt


@timeit
def train():
  sp_encoder = spm.SentencePieceProcessor(model_file=f'{ROOTDIR}/misc/encoder.model')
  sp_decoder = spm.SentencePieceProcessor(model_file=f'{ROOTDIR}/misc/decoder.model')

  model = get_model(tkn('PAD'))
  report_model_size(model)
  #optim = torchao.optim.AdamW4bit(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
  optim = _AdamW(model.parameters(), lr=LEARNING_RATE, bf16_stochastic_round=True)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=100)
  model, optim, loss = load_checkpoint(model, optim)
  data_generator = cycle(sp_encoder, sp_decoder)
  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
      src, src_mask, tgt = next(data_generator)
      loss = model(src, tgt, mask=src_mask)
      (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'{i}: {loss.item()}')
    optim.step()
    scheduler.step(i/NUM_BATCHES)
    optim.zero_grad()

    if i % CHECKPOINT_EVERY == 0:
      report_cuda_size()
      if i > 0:
        save_checkpoint(model, optim, loss)
    if i % GENERATE_EVERY == 0:
        model.eval()
        src, src_mask, tgt  = next(data_generator)
        src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
        start_tokens = torch.tensor([tkn('DECSTART')]).to('cuda')
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, eos_token=tkn('EOS'), mask = src_mask)
        print_stmt = ""
        print_stmt += f"\ninput tokenized:  \n{detokenize_bytes(sp_encoder, src.tolist()[0])} \n"
        print_stmt += f"\npredicted detokenized:  \n{detokenize_bytes(sp_decoder, sample.tolist())}\n"
        print_stmt += f"\nactual detokenized:     \n{detokenize_bytes(sp_decoder, tgt.tolist()[0])}\n"
        print(print_stmt)


def main():
    train()

if __name__ == '__main__':
  torch.set_float32_matmul_precision('medium')
  torch.backends.cudnn.benchmark = True
  torch.backends.cuda.matmul.allow_tf32 = True
  main()
