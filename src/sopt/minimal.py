import multiprocessing
from x_transformers import XTransformer
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from  subprocess import PIPE, Popen, run
import os
import random
import string
import platform
import ast
import csv
import gzip
import shutil
import time
from functools import wraps
import torch
import tqdm

ARCH = 'x86'

if platform.system() == 'Linux':
  if ARCH == 'riscv':
    CC = 'riscv64-linux-gnu-gcc'
    STRIP = 'riscv64-linux-gnu-strip'
    OBJDUMP = 'riscv64-linux-gnu-objdump'
  elif ARCH == 'x86':
    CC = 'gcc'
    STRIP = 'strip'
    OBJDUMP = 'objdump'
elif platform.system() == 'Darwin':
  if ARCH == 'riscv':
    CC = 'riscv64-elf-gcc'
    STRIP = 'riscv64-elf-strip'
    OBJDUMP = 'riscv64-elf-objdump'
  elif ARCH == 'x86':
    CC = 'x86_64-elf-gcc'
    STRIP = 'x86_64-elf-strip'
    OBJDUMP = 'x86_64-elf-objdump'
  elif ARCH == 'aarch64':
    CC = 'aarch64-elf-gcc'
    STRIP = 'aarch64-elf-strip'
    OBJDUMP = 'aarch64-elf-objdump'




GENERATE_EVERY = 1000
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)
NUM_TOKENS = 258
ENC_SEQ_LEN = 512
DEC_SEQ_LEN = 512
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
TMP = '/tmp/sopt'
DICTIONARY = f'{ROOTDIR}/misc/x86_dictionary'


def timeit(func):
  @wraps(func)
  def timeit_wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
    return result
  return timeit_wrapper

nvmlInit_called = False

def report_cuda_size():
  global nvmlInit_called
  if torch.cuda.is_available():
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    if not nvmlInit_called:
      nvmlInit()
      nvmlInit_called = True
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'cuda total    : {info.total // 1024 // 1024}MB')
    print(f'cuda free     : {info.free // 1024 // 1024}MB')
    print(f'cuda used     : {info.used // 1024 // 1024}MB')

def randstring(n):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def tkn(t):
  if isinstance(t, str):
    if t == 'PAD': return 256
    elif t == 'DECSTART': return 257
  elif isinstance(t, bytes):
    return list(t)

def tokenize(data: bytes, dictionary: str) -> bytes:
  ret = run(f"zstd -D {dictionary} --ultra -22 -c -".split(), input=data,  stdout=PIPE, stderr=PIPE)
  return ret.stdout

def detokenize(data: [int], dictionary: str) -> bytes:
  data = [x for x in data if 0 <= x <= 255]
  ret = run(f"zstd -D {dictionary} --ultra -22 -d -c -".split(), input=bytes(data),  stdout=PIPE, stderr=PIPE)
  if len(ret.stderr) > 0:
    return ret.stderr + bytes(data)
  return ret.stdout

def yarpgen(uuid: str, cc: str) -> bytes:
  yarpgen = run(f'/{ROOTDIR}/bin/{platform.system()}/yarpgen --std=c -o /{TMP}/yarpgen_{uuid}'.split(), stdout=PIPE, stderr=PIPE)
  preprocessed = run(f'{cc} -E {TMP}/yarpgen_{uuid}/func.c'.split(), stdout=PIPE, stderr=PIPE)
  return preprocessed.stdout


def compile(code: bytes, cc: str, strip: str):
  unopt_obj = f'{TMP}/{randstring(32)}.o'
  opt_obj = f'{TMP}/{randstring(32)}.o'

  unopt = run(f'{cc} -o {unopt_obj} -O0 -Wall -fcf-protection=none -xc -c -'.split(), input=code, stdout=PIPE,stderr=PIPE)
  opt = run(f'{cc} -o {opt_obj} -O3 -Wall -fcf-protection=none -xc -c -'.split(), input=code, stdout=PIPE, stderr=PIPE)

  unopt = run(f'{strip} {unopt_obj}'.split())
  opt = run(f'{strip} {opt_obj}'.split())

  # uncomment these lines to create training data for zstd dictionary
  shutil.copyfile(unopt_obj, f'/tmp/sopt/all_yarpgen/{randstring(32)}.o')
  shutil.copyfile(opt_obj, f'/tmp/sopt/all_yarpgen/{randstring(32)}.o')

  unopt_data = open(unopt_obj, 'rb').read()
  opt_data = open(opt_obj, 'rb').read()

  os.remove(unopt_obj)
  os.remove(opt_obj)

  return unopt_data, opt_data


def disasm(o_file: str, objdump: str):
  _ = Popen(f'{objdump} -M no-aliases -d {o_file}'.split(), stdout=PIPE)
  out, err = _.communicate()
  return out.decode('utf-8')


def cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len):
  if len(training_data) < batch_size:
    print("db_idx", db_idx)
    with gzip.open(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz','rt') as f:
      reader = csv.DictReader(f)
      for entry in reader:
        unopt = list(ast.literal_eval(entry['unopt']))
        opt = list(ast.literal_eval(entry['opt']))
        if len(unopt) >= ENC_SEQ_LEN:
          continue
        if len(opt) >= DEC_SEQ_LEN:
          continue
        opt.insert(0,tkn('DECSTART'))
        mask = [True]*len(unopt)
        mask.extend([False]*(encoder_len-len(unopt)))
        unopt.extend([tkn('PAD')] * (encoder_len - len(unopt)))
        opt.extend([tkn('PAD')] * (decoder_len - len(opt)))
        training_data.append([unopt, opt, mask])
      db_idx += 1
      if not os.path.exists(f'/{ROOTDIR}/data/processed_{db_idx}.csv.gz'):
        db_idx = 0
  batch = training_data[:batch_size]
  training_data = training_data[batch_size:]
  mysrc = torch.tensor(list(x[0] for x in batch)).long().to(device)
  mytgt = torch.tensor(list(x[1] for x in batch)).long().to(device)
  mysrc_mask = torch.tensor(list(x[2] for x in batch)).bool().to(device)
  return mysrc, mysrc_mask, mytgt, training_data, db_idx


ALL_INPUTS = set()
def gen(uuid):
  with open(f'/{TMP}/data/db_{uuid}.csv.gz','w+t') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writeheader()
    for x in range(100):
      if uuid == 0 and x % 10 == 0:
          print(x)
      prog = yarpgen(uuid, CC)
      unopt, opt = compile(prog, CC, STRIP)
      unopt = tokenize(unopt, f'{ROOTDIR}/misc/x86_dictionary')
      opt = tokenize(opt, f'{ROOTDIR}/misc/x86_dictionary')
      if h := hash(unopt) in ALL_INPUTS:
        continue
      ALL_INPUTS.add(h)
      writer.writerow({'c': prog, 'unopt': unopt, 'opt': opt})

def generate_database():
  ncpu = multiprocessing.cpu_count()
  os.makedirs(f'{TMP}/data', exist_ok=True)
  os.makedirs(f'{TMP}/all_yarpgen', exist_ok=True)
  for uuid in range(ncpu):
    os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
  print(f"spawning {ncpu} threads")
  ALL_INPUTS = set()
  for x in range(100):
    print('processed', x)

    with multiprocessing.Pool(ncpu) as p:
      p.map(gen, list(range(ncpu)))
    #gen(0)
    OUT = list()
    for i, gz in enumerate(os.listdir(f'/{TMP}/data/')):
      with open(f'/{TMP}/data/{gz}', 'rt') as inf:
        reader = csv.DictReader(inf)
        for row in reader:
          if h := hash(row['unopt']) in ALL_INPUTS:
            continue
          else:
            ALL_INPUTS.add(h)
            OUT.append(row)
    with gzip.open(f'/{ROOTDIR}/data/processed_{x}.csv.gz', 'w+t') as outf:
      writer = csv.DictWriter(outf, ['c', 'unopt', 'opt'])
      writer.writeheader()
      writer.writerows(OUT)


def get_model(device, pad_value, num_tokens, rank, world_size):
  if device == 'cuda':
    if '2060' in torch.cuda.get_device_name():
      dim = 512
      batch_size = 8
      generate_every = 100
      enc_depth = 4
      enc_heads = 4
      dec_depth = 4
      dec_heads = 4
      dtype = torch.float16
    elif 'V100' in torch.cuda.get_device_name():
      dim = 1024
      batch_size = 32
      generate_every = 500
      enc_depth = 8
      enc_heads = 8
      dec_depth = 8
      dec_heads = 8
      dtype = torch.float16
    elif ('4090' in torch.cuda.get_device_name() or
          'A5000' in torch.cuda.get_device_name() or
          '3090' in torch.cuda.get_device_name()):
      dim = 1024
      batch_size = 32
      generate_every = 1000
      enc_depth = 10
      enc_heads = 10
      dec_depth = 10
      dec_heads = 10
      dtype = torch.bfloat16
    elif 'A100' in torch.cuda.get_device_name():
      dim = 2048
      batch_size = 64
      generate_every = 2000
      enc_depth = 20
      enc_heads = 20
      dec_depth = 20
      dec_heads = 20
      dtype = torch.bfloat16
    else:
      assert False
  else:
    dim = 512
    batch_size = 4
    generate_every = 100
    enc_depth = 4
    enc_heads = 4
    dec_depth = 4
    dec_heads = 4
    dtype = torch.float16

  model = XTransformer(
    dim=dim,
    pad_value=pad_value,
    tie_token_emb=True,
    enc_attn_flash=True,
    dec_attn_flash=True,
    return_tgt_loss=True,
    enc_num_tokens=num_tokens,
    enc_depth=enc_depth,
    enc_heads=enc_heads,
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=num_tokens,
    dec_depth=dec_depth,
    dec_heads=dec_heads,
    dec_max_seq_len=DEC_SEQ_LEN,
    attn_num_mem_kv=16,
    num_memory_tokens=20,
    use_simple_rmsnorm=True,
    ff_no_bias=True,
    ff_swish=True,
    ff_glu=True,
    attn_kv_heads=2,
    attn_gate_values=True,
    sandwich_coef=6,
    shift_tokens=1,
    use_abs_pos_emb=False,
    rotary_xpos=True,
    attn_sparse_topk=8,
    attn_talking_heads=True,
    attn_on_attn=True,
    macaron=True,
    gate_residual=True,
    dynamic_pos_bias=True,
    dynamic_pos_bias_log_distance=True,
    resi_dual=True,
    resi_dual_scale=0.1, )

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

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"num params {params // 1024 // 1024}M {params // 1024}K ")

  return model, dtype, ENC_SEQ_LEN, DEC_SEQ_LEN, batch_size, generate_every


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

    src, src_mask, tgt, training_data, db_idx = cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len)
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
      src, src_mask, tgt, training_data, db_idx = cycle(device, training_data, db_idx, batch_size, encoder_len, decoder_len)
      src, src_mask, tgt  = src[:1], src_mask[:1], tgt[:1]
      start_tokens = torch.tensor([tkn('DECSTART')]).to(device)
      sample = model.generate(src, start_tokens, decoder_len, mask = src_mask)
      #the target output always includes the 'DECSTART' token whereas the sampled output does not
      #so shift the output left one token to delete it
      for x in range(decoder_len-1):
        tgt[0,x] = tgt[0,x+1]
      incorrects = (tgt != sample).sum()
      print_stmt = f'\nRANK: {rank} start\n'
      print_stmt += f"\ninput tokenized:  \n{detokenize(src.tolist()[0],DICTIONARY)} \n"
      print_stmt += f"\npredicted detokenized:  \n{detokenize(sample.tolist(),DICTIONARY)}\n"
      print_stmt += f"\nactual detokenized:     \n{detokenize(tgt.tolist()[0],DICTIONARY)}\n"
      print_stmt += f"\nincorrects: {incorrects}\n"
      print_stmt += f'\nRANK: {rank} end\n'
      print(print_stmt)

  if world_size > 1:
    torch.distributed.destroy_process_group()

def main():
  if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #device = 'mps'
    device = 'cpu'
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
  #generate_database()
  main()
