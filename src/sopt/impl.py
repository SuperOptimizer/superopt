import multiprocessing
import shutil

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from  subprocess import PIPE, run, Popen
import os
import sentencepiece as spm
import platform
import ast
import csv
import gzip
import base64
import shutil
import torch
import numpy as np
from x_transformers import XTransformer

from util import randstring, flatten, chunkify

ARCH = 'x86'
MODEL_SIZE = "medium"
TOKENIZER = "sentencepiece"

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 -xc'

if torch.cuda.is_available():
  DEVICE = 'cuda'
  WORLD_SIZE = torch.cuda.device_count()
  RAM_SIZE = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 // 1024 // 1024
  print(f"per GPU memory available: {RAM_SIZE}GB")
else:
  if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    RAM_SIZE = 8
    DEVICE = 'mps'
  else:
    RAM_SIZE = 24
    DEVICE = 'cpu'
  WORLD_SIZE = 1

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
TMP = '/tmp/sopt'
ZSTD_DICTIONARY = f'{ROOTDIR}/misc/zstd_x86_dictionary'
if TOKENIZER == "char":
  NUM_TOKENS = 512
elif TOKENIZER == "sentencepiece":
  NUM_TOKENS = 8192 + 2
elif TOKENIZER == "zstd":
  NUM_TOKENS = 258
elif TOKENIZER == "zstd_sentencepiece":
  NUM_TOKENS = 4096 + 2

ENC_SEQ_LEN = 2048
DEC_SEQ_LEN = 2048
GENERATE_EVERY = 100
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)

if torch.cuda.is_available():
  DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] > 7 else torch.float16
  CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'
else:
  DTYPE = torch.float16
  CHECKPOINT = f'/{ROOTDIR}/checkpoint-cpu-{MODEL_SIZE}.pt'

if platform.system() == 'Linux':
  if ARCH == 'riscv':
    GCC = 'riscv64-linux-gnu-gcc'
    STRIP = 'riscv64-linux-gnu-strip'
    OBJDUMP = 'riscv64-linux-gnu-objdump'
  elif ARCH == 'x86':
    GCC = 'gcc'
    CLANG = 'clang-18'
    STRIP = 'strip'
    OBJDUMP = 'objdump'
    OBJCOPY = 'objcopy'
elif platform.system() == 'Darwin':
  if ARCH == 'riscv':
    GCC = 'riscv64-elf-gcc'
    STRIP = 'riscv64-elf-strip'
    OBJDUMP = 'riscv64-elf-objdump'
  elif ARCH == 'x86':
    GCC = 'x86_64-elf-gcc'
    STRIP = 'x86_64-elf-strip'
    OBJDUMP = 'x86_64-elf-objdump'
    OBJCOPY = 'x86_64-elf-objcopy'
  elif ARCH == 'aarch64':
    GCC = 'aarch64-elf-gcc'
    STRIP = 'aarch64-elf-strip'
    OBJDUMP = 'aarch64-elf-objdump'


def get_model(rank, pad_value):
  size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[MODEL_SIZE]

  model = XTransformer(
    dim=[256,512,768,1024][size],
    pad_value=pad_value,
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

    enc_attn_num_mem_kv=[6,12,18,24][size],
    enc_num_memory_tokens=[6,12,18,24][size],
    enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    enc_ff_swish=True,
    enc_ff_glu=True,
    enc_attn_kv_heads=[1,2,3,4][size],
    enc_attn_gate_values=True,
    enc_sandwich_coef=[2,4,6,8][size],
    enc_shift_tokens=1,
    enc_use_abs_pos_emb=False,
    enc_attn_on_attn=True,
    enc_macaron=True,
    enc_resi_dual=True,
    enc_resi_dual_scale=0.1,

    dec_attn_num_mem_kv=[6,12,18,24][size],
    dec_num_memory_tokens=[6,12,18,24][size],
    dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    dec_ff_swish=True,
    dec_ff_glu=True,
    dec_attn_kv_heads=[1,2,3,4][size],
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
    if '2060' not in torch.cuda.get_device_name():
      model = torch.compile(model)
  else:
    model = model.to(DEVICE)

  if RAM_SIZE <= 8:
    batch_size = {"small": 8, "medium": 1, "large": 0, "xl": 0}[MODEL_SIZE]
  elif RAM_SIZE <= 16:
    batch_size = {"small": 8, "medium": 2, "large": 0, "xl": 0}[MODEL_SIZE]
  elif RAM_SIZE <= 24:
    batch_size = {"small": 40, "medium": 10, "large": 3, "xl": 1}[MODEL_SIZE]
  elif RAM_SIZE <= 48:
    batch_size = {"small": 8, "medium": 2, "large": 0, "xl": 0}[MODEL_SIZE]
  elif RAM_SIZE <= 80:
    batch_size = {"small": 8, "medium": 2, "large": 0, "xl": 0}[MODEL_SIZE]
  return model, batch_size

def tkn_sp(t):
  if t == 'DECSTART':
    return 8192
  elif t == 'PAD':
    return 8193
  assert False

def tkn_char(t):
  if t == 'PAD':
    return 256
  elif t == 'DECSTART':
    return 257
  assert False

def zstd_train():
  os.makedirs(f'/{TMP}/all_objs', exist_ok=True)
  for db_idx in range(len(os.listdir(f'/{ROOTDIR}/cleandata/')))[:25]:
    with gzip.open(f'/{ROOTDIR}/cleandata/processed_{db_idx}.csv.gz', 'rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = ast.literal_eval(entry['unopt'])
        opt = ast.literal_eval(entry['opt'])
        with open(f'/{TMP}/all_objs/{randstring(16)}.o','w+b') as outf:
          outf.write(unopt)
        with open(f'/{TMP}/all_objs/{randstring(16)}.o','w+b') as outf:
          outf.write(opt)

def sentencepiece_train(zstd=False):
  with (open(f'{TMP}/sentencepiece_encoder.txt', 'w+t') as encoder_f,
        open(f'{TMP}/sentencepiece_decoder.txt', 'w+t') as decoder_f):
    for db in os.listdir(f'/{ROOTDIR}/cleandata/')[:10]:
      with gzip.open(f'/{ROOTDIR}/cleandata/{db}', 'rt') as f:
        for entry in chunkify(f.readlines(),2):
          unopt = ast.literal_eval(entry[0])
          opt = ast.literal_eval(entry[1])
          if zstd:
            unopt = zstd_compress(unopt)
            opt = zstd_compress(opt)
          encoder_f.write(base64.b64encode(unopt).decode('utf-8') + '\n')
          decoder_f.write(base64.b64encode(opt).decode('utf-8') + '\n')



sp = None

def tokenize_zstdsp(data: bytes):
  global sp
  if sp is None:
    sp = spm.SentencePieceProcessor()
    sp.load(f'{ROOTDIR}/misc/x86_zstd_sp8k.model')
  return sp.encode(base64.b64encode(zstd_compress(data)))

def detokenize_zstdsp(tokens: [int]):
  global sp
  if sp is None:
    sp = spm.SentencePieceProcessor()
    sp.load(f'{ROOTDIR}/misc/x86_zstd_sp8k.model')
  tokens = [t for t in tokens if t < NUM_TOKENS-2]
  tokens = sp.decode(tokens)
  try:
    tokens = base64.b64decode(tokens)
    tokens = zstd_decompress(tokens)
  except:
    tokens = "invalid".encode('utf-8')
  return tokens

def tokenize_zstd(data:bytes):
  return zstd_compress(data)

def detokenize_zstd(data:[int]):
  data = [x for x in data if 0 <= x <= 255]
  return zstd_decompress(bytes(data))

def tokenize_sp(data: bytes):
  global sp
  if sp is None:
    sp = spm.SentencePieceProcessor()
    sp.load(f'{ROOTDIR}/misc/x86_sp8k.model')
  tokens = sp.encode(base64.b64encode(data).decode('utf-8'))
  return tokens


def detokenize_sp(tokens: [int]):
  global sp
  if sp is None:
    sp = spm.SentencePieceProcessor()
    sp.load(f'{ROOTDIR}/misc/x86_sp8k.model')
  tokens = [t for t in tokens if t < NUM_TOKENS-2]
  tokens = sp.decode(tokens)
  try:
    tokens = base64.b64decode(tokens)
  except:
    tokens = "invalid".encode('utf-8')
  return tokens

def tokenize_char(data: bytes):
  '''0-255 encode that data
     256 = PAD
     257 = DECSTART
     258 - 511 = tokenval - 2 zeroes in a row
  '''
  ret = []
  nzero = 0
  for b in list(data):
    if b != 0:
      if nzero == 0:
        ret.append(b)
      elif nzero == 1:
        ret.append(0)
        ret.append(b)
        nzero = 0
      else:
        ret.append(nzero + 256)
        ret.append(b)
        nzero = 0
    else:
      if nzero == 254:
        ret.append(511)
        nzero = 0
      else:
        nzero +=1
  if nzero > 0:
    ret.append(nzero + 256)
  return ret

def detokenize_char(tokens: [int]):
  ret = []
  for t in tokens:
    if 0 <= t <= 255:
      ret.append(t)
    elif 258 <= t <= 511:
      ret.extend([0] * (t - 256))
    else:
      pass #no need to pass meta tokens to detokenize
  return bytes(ret)

def zstd_compress(data: bytes) -> bytes:
  ret = run(f"zstd -D {ZSTD_DICTIONARY} --ultra -22 -c -".split(), input=data,  stdout=PIPE, stderr=PIPE)
  return ret.stdout

def zstd_decompress(data: bytes) -> bytes:
  ret = run(f"zstd -D {ZSTD_DICTIONARY} --ultra -22 -d -c -".split(), input=bytes(data),  stdout=PIPE, stderr=PIPE)
  if len(ret.stderr) > 0:
    return ret.stderr + bytes(data)
  return ret.stdout

def gen_yarpgen(uuid):
  ret = []
  for x in range(100):
    if uuid == 0 and x % 10 == 0:
      print(x)
    yarpgen = run(f'/{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c -o /{TMP}/yarpgen_{uuid}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    preprocessed = run(f'{GCC} -E -xc /{TMP}/yarpgen_{uuid}/func.c'.split(),stdin=PIPE,stdout=PIPE,stderr=PIPE)
    prog = preprocessed.stdout.decode('utf-8')
    prog = ' '.join(line for line in prog.split('\n') if not line.startswith('#'))
    ret.append(prog)
  return ret


def compile(txt_gz):
  print(f"processing {txt_gz}")
  ret = []
  i = 0
  with gzip.open(f'/{ROOTDIR}/yarpgen/{txt_gz}', 'rt') as f:
    for prog in f:
      i += 1
      print(f"processed {i}")

      unopt_o = f'/{TMP}/{randstring(32)}.o'
      opt_o = f'/{TMP}/{randstring(32)}.o'

      clang_unopt_o = f'/{TMP}/{randstring(32)}.o'
      clang_opt_o = f'/{TMP}/{randstring(32)}.o'

      unopt_cc_gcc = Popen(f'{GCC} -o {unopt_o} -O0 {CCFLAGS} -xc -c -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_cc_gcc = Popen(f'{GCC} -o {opt_o} -O3 {CCFLAGS} -xc -c -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_cc_clang = Popen(f'{CLANG} -o {clang_unopt_o} -O0 {CCFLAGS} -xc -c -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_cc_clang = Popen(f'{CLANG} -o {clang_opt_o} -O3 {CCFLAGS} -xc -c -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      unopt_cc_gcc.communicate(prog.encode('utf-8'))
      opt_cc_gcc.communicate(prog.encode('utf-8'))
      unopt_cc_clang.communicate(prog.encode('utf-8'))
      opt_cc_clang.communicate(prog.encode('utf-8'))

      unopt_strip_gcc = Popen(f'{STRIP} {unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_strip_gcc = Popen(f'{STRIP} {opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_strip_clang = Popen(f'{STRIP} {clang_unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_strip_clang = Popen(f'{STRIP} {clang_opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      unopt_strip_gcc.communicate()
      opt_strip_gcc.communicate()
      unopt_strip_clang.communicate()
      opt_strip_clang.communicate()

      unopt_objcopy_gcc = Popen(f'{OBJCOPY} --remove-section .comment {unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_objcopy_gcc = Popen(f'{OBJCOPY} --remove-section .comment {opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_objcopy_clang = Popen(f'{OBJCOPY} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {clang_unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_objcopy_clang = Popen(f'{OBJCOPY} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {clang_opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      unopt_objcopy_gcc.communicate()
      opt_objcopy_gcc.communicate()
      unopt_objcopy_clang.communicate()
      opt_objcopy_clang.communicate()

      with open(unopt_o, 'rb') as f:
        unopt_gcc = f.read()
      with open(opt_o, 'rb') as f:
        opt_gcc = f.read()

      with open(clang_unopt_o, 'rb') as f:
        unopt_clang = f.read()
      with open(clang_opt_o, 'rb') as f:
        opt_clang = f.read()

      if len(unopt_gcc) > 50000 or len(opt_gcc) > 50000:
        print(f"skipping too long prog {len(unopt_gcc)} {len(opt_gcc)}")
      else:
        ret.append({'unopt':unopt_gcc, 'opt': opt_gcc})

      if len(unopt_clang) > 50000 or len(opt_clang) > 50000:
        print(f"skipping too long prog {len(unopt_clang)} {len(opt_clang)}")
      else:
        if not (unopt_clang == unopt_gcc and opt_clang == opt_gcc):
          ret.append({'unopt':unopt_clang, 'opt': opt_clang})

      os.remove(unopt_o)
      os.remove(opt_o)
      os.remove(clang_unopt_o)
      os.remove(clang_opt_o)
  return txt_gz, ret


def generate_yarpgen():
  print("generating yarpgen")
  n_preexisting = len(os.listdir(f'{ROOTDIR}/yarpgen/'))
  ncpu = multiprocessing.cpu_count()
  for uuid in range(ncpu):
    os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
  for x in range(100):
    print('processed', x)
    with multiprocessing.Pool(ncpu) as p:
      ret = p.map(gen_yarpgen, list(range(ncpu)))
    with gzip.open(f'{ROOTDIR}/yarpgen/{x + n_preexisting}.txt.gz', 'w+t') as outf:
      for line in flatten(ret):
        outf.write(line + '\n')

def compile_yarpgen():
  ncpu = multiprocessing.cpu_count()
  preexisting = os.listdir(f'{ROOTDIR}/rawdata/')
  os.makedirs(f'{ROOTDIR}/rawdata', exist_ok=True)
  args = []
  for txt_gz in sorted(os.listdir(f'{ROOTDIR}/yarpgen')):
    if txt_gz not in preexisting:
      args.append(txt_gz)
  for chunk in chunkify(args,ncpu):
    with multiprocessing.Pool(ncpu) as p:
      for ret in p.map(compile, chunk):
        idx,listings = ret
        with gzip.open(f'{ROOTDIR}/rawdata/{idx}', 'wt') as outf:
          for l in listings:
            outf.write(str(l['unopt']) + '\n')
            outf.write(str(l['opt']) + '\n')

def clean_yarpgen():
  all_programs = set()
  for txt_gz in sorted(os.listdir(f'{ROOTDIR}/rawdata')):
    with gzip.open(f'{ROOTDIR}/rawdata/{txt_gz}', 'rt') as inf, gzip.open(f'{ROOTDIR}/cleandata/{txt_gz}', 'w+t') as outf:
      print(f"processing {txt_gz}")
      for line in chunkify(inf.readlines(),2):
        unopt,opt = line
        if (h := hash(unopt)) in all_programs:
          print("skipping prog")
          continue
        all_programs.add(h)
        outf.write(unopt)
        outf.write(opt)
      #reader = csv.DictReader(inf)
      #writer = csv.DictWriter(outf, ['unopt','opt'])
      #for row in reader:
      #  if (h := hash(row['unopt'])) in all_programs:
      #    continue
      #  all_programs.add(h)
      #  writer.writerow(row)
      #  print()

def save_checkpoint(model,  optim, loss, scaler, scheduler):
  if DEVICE == 'cuda':
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