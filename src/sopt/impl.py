import torch.utils.data
from  subprocess import PIPE, run, Popen
import os
import sentencepiece as spm
import platform
import torch
from x_transformers import XTransformer
import x_transformers
import torchao
import gzip
from torchao.float8 import convert_to_float8_training

import torch
import torch.nn as nn
from torch.nn import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration, T5Config

ARCH = 'x86'
MODEL_SIZE = "small"

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 '

DEVICE = 'cuda'
RAM_SIZE = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 // 1024 // 1024

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
TMP = '/tmp/sopt'

#vocab tokens are the first 0 through NUM_VOCAB_TOKENS-1, used by sentencepiece
NUM_VOCAB_TOKENS = 4094
NUM_SPECIAL_TOKENS = 2
NUM_TOKENS = NUM_VOCAB_TOKENS + NUM_SPECIAL_TOKENS

ENC_SEQ_LEN = 4096
DEC_SEQ_LEN = 4096
GENERATE_EVERY = 1000
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)
BATCH_SIZE = 1

DTYPE = torch.bfloat16
CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

GCC = 'gcc'
CLANG = 'clang'
CLANGPP = 'clang++'
STRIP = 'strip'
OBJDUMP = 'objdump'
OBJCOPY = 'objcopy'

def get_model(pad_value):
  size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[MODEL_SIZE]
  model = XTransformer(
    dim=[256, 512, 768, 1024][size],
    pad_value=pad_value,
    tie_token_emb=True,
    enc_attn_flash=True,
    dec_attn_flash=True,
    return_tgt_loss=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=4,
    enc_heads=4,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=4,
    dec_heads=4,
    dec_max_seq_len=DEC_SEQ_LEN,

    #enc_attn_num_mem_kv=[6, 12, 18, 24][size],
    #enc_num_memory_tokens=[6, 12, 18, 24][size],
    #enc_use_simple_rmsnorm=True,
    #enc_ff_no_bias=True,
    #enc_ff_swish=True,
    #enc_ff_glu=True,
    #enc_attn_kv_heads=[1, 2, 3, 4][size],
    #enc_attn_gate_values=True,
    #enc_sandwich_coef=[2, 4, 6, 8][size],
    #enc_shift_tokens=1,
    #enc_use_abs_pos_emb=False,
    #enc_attn_on_attn=True,
    #enc_macaron=True,
    # enc_rotary_pos_emb=True,
    #enc_alibi_pos_bias=True,
    #enc_alibi_num_heads=[2, 4, 6, 8][size],

    #dec_attn_num_mem_kv=[6, 12, 18, 24][size],
    #dec_num_memory_tokens=[6, 12, 18, 24][size],
    #dec_use_simple_rmsnorm=True,
    #dec_ff_no_bias=True,
    #dec_ff_swish=True,
    #dec_ff_glu=True,
    #dec_attn_kv_heads=[1, 2, 3, 4][size],
    #dec_attn_gate_values=False,
    #dec_sandwich_coef=[2, 4, 6, 8][size],
    #dec_shift_tokens=1,
    #dec_use_abs_pos_emb=False,
    #dec_attn_on_attn=False,
    #dec_macaron=False,
    # dec_rotary_pos_emb=True,
    #dec_alibi_pos_bias=False,
    #dec_alibi_num_heads=[2, 4, 6, 8][size])
  )


  model = model.cuda()
  #model = torch.compile(model)

  return model

# our tokenization scheme is
# byte -> [0-9A-F][0-9A-F]
# where each byte in the stream is mapped to the high,low nibble text hex representation
# this then gets fed into sentencepiece for the final vocab
def bytes_to_hex_string(arr: bytes):
  return arr.hex().upper()

def hex_string_to_bytes(hex_string):
  try:
    return bytes.fromhex(hex_string)
  except:
    return b"invalid"

def tkn(str):
  if str == 'PAD':
    return NUM_VOCAB_TOKENS + 0
  elif str == 'DECSTART':
    return NUM_VOCAB_TOKENS + 1
  raise

def tokenize_bytes(sp, data: bytes):
  tokens = sp.encode(bytes_to_hex_string(data))
  return tokens

def detokenize_bytes(sp, tokens: [int]):
  tokens = [t for t in tokens if t < NUM_VOCAB_TOKENS]
  hexstr = sp.decode(tokens)
  return hex_string_to_bytes(hexstr)

def tokenize_hexstr(sp, data: str):
  tokens = sp.encode(data)
  return tokens

def detokenize_hexstr(sp, tokens: [int]):
  tokens = [t for t in tokens if t < NUM_VOCAB_TOKENS]
  hexstr = sp.decode(tokens)
  return hexstr

def gen_yarpgen(threadnum, num):
  yarpgen = f'/{ROOTDIR}/bin/{platform.system()}/yarpgen'
  outdir = f'/{TMP}/yarpgen_{threadnum}'
  os.makedirs(outdir, exist_ok=True)
  c_file = f'{outdir}/func.c'
  opt_obj = f'{outdir}/func.opt.o'
  unopt_obj = f'{outdir}/func.unopt.o'

  for x in range(num):
    print(x)
    ret = run(f'{yarpgen} --std=c -o {outdir}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang -c {c_file} -o {unopt_obj} -include stdint.h -O0 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'clang -c {c_file} -o {opt_obj}   -include stdint.h -O3 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'{OBJCOPY}  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    ret = run(f'{OBJCOPY}  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if ret.returncode != 0:
      raise
    with open(unopt_obj, 'rb') as f, open(opt_obj, 'rb') as g:
      ret=(f.read(),g.read())
      yield ret

def gen_model_training_data():
  os.makedirs(TMP, exist_ok=True)
  progs = gen_yarpgen(0, 2000)

  encoder_corpus = f"{TMP}/encoder.txt.gzip"
  decoder_corpus = f"{TMP}/decoder.txt.gzip"

  with gzip.open(encoder_corpus, 'at') as f, gzip.open(decoder_corpus, 'at') as g:
    for pair in progs:
      unopt, opt = pair
      f.write(bytes_to_hex_string(unopt) + "\n")
      g.write(bytes_to_hex_string(opt) + "\n")

def gen_sentencepiece_training_data():
  os.makedirs(TMP, exist_ok=True)
  progs = gen_yarpgen(0,2000)

  encoder_corpus = f"{TMP}/encoder.txt"
  decoder_corpus = f"{TMP}/decoder.txt"

  with open(encoder_corpus, 'at') as f, open(decoder_corpus, 'at') as g:
    for pair in progs:
      unopt,opt = pair
      f.write(bytes_to_hex_string(unopt) + "\n")
      g.write(bytes_to_hex_string(opt) + "\n")
  #spm_train --input=encoder.txt --model_prefix=encoder --vocab_size=4096 --max_sentence_length=655350 --character_coverage=1.0 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false
  #run(f"spm_train --input={encoder_corpus} --model_prefix=encoder --vocab_size=8192 --character_coverage=1.0 --model_type=unigram --max_sentence_length=65535 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false".split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,cwd=TMP)
  #run(f"spm_train --input={decoder_corpus} --model_prefix=decoder --vocab_size=8192 --character_coverage=1.0 --model_type=unigram --max_sentence_length=65535 --bos_id=-1 --eos_id=-1 --pad_id=-1  --add_dummy_prefix=false --split_by_number=false".split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,cwd=TMP)


def save_checkpoint(model,  optim, loss, scaler, scheduler):
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss.item(),
    'scaler': scaler.state_dict(),
    'scheduler': scheduler.state_dict()},
    CHECKPOINT)

def load_checkpoint(model, optim, loss):
  if os.path.exists(CHECKPOINT):
    print(f"loading {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
  return model, optim,  loss


def gen_model_training_data_parallel():
  import concurrent.futures
  import multiprocessing

  # Determine available CPU threads
  num_threads = multiprocessing.cpu_count()

  # Total programs to generate
  total_programs = 20000

  # Split workload across threads
  programs_per_thread = total_programs // num_threads
  remainder = total_programs % num_threads

  os.makedirs(TMP, exist_ok=True)

  # Function to generate programs for a specific thread
  def generate_for_thread(thread_id):
    num_programs = programs_per_thread + (1 if thread_id < remainder else 0)
    temp_encoder_file = f"{TMP}/encoder_{thread_id}.txt.gzip"
    temp_decoder_file = f"{TMP}/decoder_{thread_id}.txt.gzip"

    with gzip.open(temp_encoder_file, 'wt') as f, gzip.open(temp_decoder_file, 'wt') as g:
      for pair in gen_yarpgen(thread_id, num_programs):
        unopt, opt = pair
        f.write(bytes_to_hex_string(unopt) + "\n")
        g.write(bytes_to_hex_string(opt) + "\n")

    return (temp_encoder_file, temp_decoder_file)

  # Create output files
  encoder_corpus = f"{TMP}/encoder.txt.gzip"
  decoder_corpus = f"{TMP}/decoder.txt.gzip"

  # Run tasks in parallel using threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(generate_for_thread, i) for i in range(num_threads)]
    temp_files = [future.result() for future in concurrent.futures.as_completed(futures)]

  # Combine all temporary files into final output
  with gzip.open(encoder_corpus, 'at') as f_enc, gzip.open(decoder_corpus, 'at') as f_dec:
    for enc_file, dec_file in temp_files:
      with gzip.open(enc_file, 'rt') as temp_enc, gzip.open(dec_file, 'rt') as temp_dec:
        f_enc.write(temp_enc.read())
        f_dec.write(temp_dec.read())

      # Cleanup temporary files
      os.remove(enc_file)
      os.remove(dec_file)

#gen_model_training_data_parallel()
#gen_sentencepiece_training_data()
#gen_model_training_data()