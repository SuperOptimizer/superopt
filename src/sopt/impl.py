import multiprocessing
import shutil
import math
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.utils.data
from  subprocess import PIPE, run, Popen
import os
import sentencepiece as spm
import platform
import multiprocessing.dummy
import ast
import csv
import gzip

from torch.utils.data import DataLoader
import base64
import shutil
import torch
import numpy as np
from x_transformers import XTransformer
import lightning as L

from util import randstring, flatten, chunkify

ARCH = 'x86'
MODEL_SIZE = "medium"
TOKENIZER = "sentencepiece"

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 '

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
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
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
GENERATE_EVERY = 1000
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)
BATCH_SIZE = 1

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
    CLANGPP = 'clang++-18'
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
    batch_size = {"small": 40, "medium": 8, "large": 3, "xl": 1}[MODEL_SIZE]
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
  for txt_gz in os.listdir(f'/{ROOTDIR}/cleandata/')[:1]:
    with gzip.open(f'/{ROOTDIR}/cleandata/{txt_gz}', 'rt') as f:
      for entry in chunkify(f.readlines()[:25000],2):
        unopt = ast.literal_eval(entry[0])
        opt = ast.literal_eval(entry[1])
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
  print("gen yarpgen")
  ret = []
  for x in range(25):
    if uuid == 0 and x % 10 == 0:
      print(x)
    yarpgen = run(f'/{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c++ -o /{TMP}/yarpgen_{uuid}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    with open(f'/{TMP}/yarpgen_{uuid}/func.cpp', 'rt') as f:
      prog = f.read()
    with open(f'/{TMP}/yarpgen_{uuid}/init.h', 'rt') as f:
      prog = f.read() + '\n' + prog
    newprog = ['/*lang=c++*/']
    for line in prog.split('\n'):
      if '#include "init.h"' not in line:
        newprog.append(line)
    prog = '\n'.join(newprog)
    ret.append(base64.b64encode(prog.encode('utf-8')))
  return ret

def gen_csmith(uuid):
  print("gen csmith")
  ret = []
  for x in range(25):
    if uuid == 0 and x % 10 == 0:
      print(x)
    csmith = run(f'/{ROOTDIR}/bin/{platform.system()}/csmith --concise --max-funcs 1 --no-safe-math --nomain'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    prog = csmith.stdout.decode('utf-8')
    newprog = ['/*lang=c*/\n']
    for line in prog.split('\n'):
      if '#include "csmith.h' not in line:
        newprog.append(line)
    prog = '\n'.join(newprog)
    ret.append(base64.b64encode(prog.encode('utf-8')))
  return ret

def gen_ldrgen(uuid):
  print("gen ldrgen")
  #note: expects ocaml frama-c and ldrgen plugin to already be installed
  #frama-c -ldrgen -ldrgen-int-only
  ret = []
  for x in range(25):
    if uuid == 0 and x % 10 == 0:
      print(x)
    ldrgen = run(f'/{HOMEDIR}/.opam/4.14.1/bin/frama-c -ldrgen -ldrgen-int-only'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    prog = ldrgen.stdout.decode('utf-8')
    prog = '/*lang=c*/\n'
    ret.append(base64.b64encode(prog.encode('utf-8')))
  return ret

def gen_ccg(uuid):
  print("gen ccg")
  ret = []
  for x in range(25):
    if uuid == 0 and x % 10 == 0:
      print(x)
    ccg = run(f'/{ROOTDIR}/bin/{platform.system()}/ccg --max-function 1 --max-localvars 4 --max-function-parameters 8 --min-statements-per-block 1 --max-statements-per-block 4 --max-expression-nesting 4 --max-block-nesting 4'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    prog = ccg.stdout.decode('utf-8')
    newprog = ['/*lang=c*/\n']
    for line in prog.split('\n'):
      if 'int main' in line:
        break
      newprog.append(line)
    prog = '\n'.join(newprog)
    ret.append(base64.b64encode(prog.encode('utf-8')))
  return ret

def compile(txt_gz):
  print(f"processing {txt_gz}")
  ret = []
  i = 0
  with gzip.open(f'/{ROOTDIR}/randomprograms/{txt_gz}', 'rt') as f:
    for prog in f:
      prog = base64.b64decode(ast.literal_eval(prog)).decode('utf-8')
      i += 1
      print(f"processed {i}")

      unopt_o = f'/{TMP}/{randstring(32)}.o'
      opt_o = f'/{TMP}/{randstring(32)}.o'

      clang_unopt_o = f'/{TMP}/{randstring(32)}.o'
      clang_opt_o = f'/{TMP}/{randstring(32)}.o'

      clang = f'{CLANGPP} -xc++ -stdlib=libc++ ' if 'lang=c++' in prog else f'{CLANG} -xc '
      gcc = 'g++ -xc++ ' if 'lang=c++' in prog else 'gcc -xc '

      unopt_cc_gcc = Popen(f'{gcc} -o {unopt_o} -O0 {CCFLAGS} -c -include stdint.h -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_cc_gcc = Popen(f'{gcc} -o {opt_o} -O3 {CCFLAGS} -c -include stdint.h -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_cc_clang = Popen(f'{clang} -o {clang_unopt_o} -O0 {CCFLAGS}  -c -include stdint.h -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_cc_clang = Popen(f'{clang} -o {clang_opt_o} -O3 {CCFLAGS}  -c -include stdint.h -'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      out1,err1 = unopt_cc_gcc.communicate(prog.encode('utf-8'))
      out2,err2 = opt_cc_gcc.communicate(prog.encode('utf-8'))
      out3,err3 = unopt_cc_clang.communicate(prog.encode('utf-8'))
      out4,err4 = opt_cc_clang.communicate(prog.encode('utf-8'))

      unopt_strip_gcc = Popen(f'{STRIP} {unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_strip_gcc = Popen(f'{STRIP} {opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_strip_clang = Popen(f'{STRIP} {clang_unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_strip_clang = Popen(f'{STRIP} {clang_opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      out5,err5 = unopt_strip_gcc.communicate()
      out6,err6 = opt_strip_gcc.communicate()
      out7,err7 = unopt_strip_clang.communicate()
      out8,err8 = opt_strip_clang.communicate()

      unopt_objcopy_gcc = Popen(f'{OBJCOPY} --remove-section .comment {unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_objcopy_gcc = Popen(f'{OBJCOPY} --remove-section .comment {opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      unopt_objcopy_clang = Popen(f'{OBJCOPY} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {clang_unopt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
      opt_objcopy_clang = Popen(f'{OBJCOPY} --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {clang_opt_o}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

      out9,err9 = unopt_objcopy_gcc.communicate()
      out10,err10 = opt_objcopy_gcc.communicate()
      out11,err11 = unopt_objcopy_clang.communicate()
      out12,err12 = opt_objcopy_clang.communicate()

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
        ret.append({'unopt':unopt_clang, 'opt': opt_clang})

      os.remove(unopt_o)
      os.remove(opt_o)
      os.remove(clang_unopt_o)
      os.remove(clang_opt_o)
  return txt_gz, ret


def generate_random_code():
  print("generating random code db")
  n_preexisting = len(os.listdir(f'{ROOTDIR}/randomcode/'))
  ncpu = multiprocessing.cpu_count()
  for uuid in range(ncpu):
    os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
  for x in range(100):
    print('processed', x)
    with multiprocessing.Pool(ncpu) as p:
      ret = p.map(gen_yarpgen, list(range(ncpu)))
    with gzip.open(f'{ROOTDIR}/randomprograms/{x + n_preexisting}.txt.gz', 'w+t') as outf:
      for line in flatten(ret):
        outf.write(line + '\n')


def generate_yarpgen():
  print("generating randomprograms")
  n_preexisting = len(os.listdir(f'{ROOTDIR}/randomprograms/'))
  ncpu = multiprocessing.cpu_count()
  for uuid in range(ncpu):
    os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
  for x in range(100):
    print('processed', x)
    with multiprocessing.Pool(ncpu) as p:
      yarpret = p.map(gen_yarpgen, list(range(ncpu)))
      ccgpret = p.map(gen_ccg, list(range(ncpu)))
      csmithret = p.map(gen_csmith, list(range(ncpu)))
      ldrgenret = p.map(gen_ldrgen, list(range(ncpu)))
    with gzip.open(f'{ROOTDIR}/randomprograms/{x + n_preexisting}.txt.gz', 'w+t') as outf:
      print(f"writing {ROOTDIR}/randomprograms/{x + n_preexisting}.txt.gz")
      for line in flatten(ccgpret) + flatten(csmithret) + flatten(yarpret) + flatten(ldrgenret):
        outf.write(str(line) + '\n')

def compile_yarpgen():
  ncpu = multiprocessing.cpu_count()
  preexisting = os.listdir(f'{ROOTDIR}/rawdata/')
  os.makedirs(f'{ROOTDIR}/rawdata', exist_ok=True)
  args = []
  for txt_gz in sorted(os.listdir(f'{ROOTDIR}/randomprograms')):
    if txt_gz not in preexisting:
      args.append(txt_gz)
  for chunk in chunkify(args,ncpu):
    with multiprocessing.dummy.Pool(ncpu) as p:
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


class SuperOptimizer(L.LightningModule):
  def __init__(self, size, pad_value):
    super().__init__()
    self.learning_rate = LEARNING_RATE
    size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[size]
    self.model = XTransformer(
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
      #enc_rotary_pos_emb=True,
      enc_alibi_pos_bias=True,
      enc_alibi_num_heads=[2,4,6,8][size],

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
      #dec_rotary_pos_emb=True,
      dec_alibi_pos_bias=True,
      dec_alibi_num_heads=[2,4,6,8][size])

  def training_step(self, batch, batch_idx):
    src,mask,tgt = batch
    loss = self.model(src, tgt, mask=mask)
    print(loss.item())
    return loss

  def configure_optimizers(self):
    optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optim

class MyDataset(torch.utils.data.IterableDataset):
  def __init__(self,path):
    self.path = path

  def generate(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      for txtgz in sorted(os.listdir(self.path)):
        training_data = []
        print(f"loading {self.path}/{txtgz}")
        with gzip.open(f'{self.path}/{txtgz}', 'rt') as f:
          asdf = f.readlines()
          for entry in chunkify(asdf, 2):
            unopt = ast.literal_eval(entry[0])
            opt = ast.literal_eval(entry[1])
            unopt_tokens = tokenize_sp(unopt)
            opt_tokens = tokenize_sp(opt)
            # print(f"len unopt tokens {len(unopt_tokens)} len opt tokens {len(opt_tokens)} len unopt {len(unopt)} len opt {len(opt)}")
            if len(unopt_tokens) >= ENC_SEQ_LEN or len(opt_tokens) >= DEC_SEQ_LEN:
              continue
            opt_tokens.insert(0, tkn_sp('DECSTART'))
            mask = [True] * len(unopt_tokens)
            mask.extend([False] * (ENC_SEQ_LEN - len(unopt_tokens)))
            unopt_tokens.extend([tkn_sp('PAD')] * (ENC_SEQ_LEN - len(unopt_tokens)))
            opt_tokens.extend([tkn_sp('PAD')] * (DEC_SEQ_LEN - len(opt_tokens)))
            training_data.append([unopt_tokens, opt_tokens, mask])
        for thing in training_data:
          mysrc = torch.tensor(thing[0]).long()
          mytgt = torch.tensor(thing[1]).long()
          mysrc_mask = torch.tensor(thing[2]).bool()
          yield mysrc, mysrc_mask, mytgt
    else:
      print()

  def __iter__(self):
    return iter(self.generate())

if __name__ == '__main__':
  from lightning.pytorch.callbacks import StochasticWeightAveraging
  from lightning.pytorch.tuner import Tuner
  devices = torch.cuda.device_count()
  if '2060' in torch.cuda.get_device_name():
    precision = "16-mixed"
  elif 'H100' in torch.cuda.get_device_name():
    precision = "transformer-engine"
  else:
    precision = "bf16-mixed"


  model = SuperOptimizer("small",tkn_sp('PAD'))
  dataset = MyDataset(f'{ROOTDIR}/rawdata')
  train_loader = DataLoader(dataset, batch_size=2)
  trainer = L.Trainer(profiler="simple",
                      fast_dev_run=100,
                      accelerator="gpu",
                      devices=devices,
                      precision=precision,
                      accumulate_grad_batches=4,
                      gradient_clip_algorithm="value",
                      callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
  trainer.fit(model=model, train_dataloaders=train_loader)