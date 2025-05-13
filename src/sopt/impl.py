import torch.utils.data
from  subprocess import PIPE, run, Popen
import os
import sentencepiece as spm
import platform
import torch
from x_transformers import XTransformer


ARCH = 'x86'
MODEL_SIZE = "small"

CCFLAGS = '-Wall -fcf-protection=none -fno-asynchronous-unwind-tables -fno-unwind-tables -march=znver3 '

DEVICE = 'cuda'
RAM_SIZE = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 // 1024 // 1024

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
TMP = '/tmp/sopt'
NUM_TOKENS = 8192 + 2

ENC_SEQ_LEN = 2048
DEC_SEQ_LEN = 2048
GENERATE_EVERY = 1000
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e5)
BATCH_SIZE = 1

DTYPE = torch.bfloat16
CHECKPOINT = f'/{ROOTDIR}/checkpoint-{torch.cuda.get_device_name()}-{MODEL_SIZE}.pt'

GCC = 'gcc'
CLANG = 'clang-18'
CLANGPP = 'clang++-18'
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
    enc_depth=[4, 8, 12, 16][size],
    enc_heads=[4, 8, 12, 16][size],
    enc_max_seq_len=ENC_SEQ_LEN,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=[4, 8, 12, 16][size],
    dec_heads=[4, 8, 12, 16][size],
    dec_max_seq_len=DEC_SEQ_LEN,

    enc_attn_num_mem_kv=[6, 12, 18, 24][size],
    enc_num_memory_tokens=[6, 12, 18, 24][size],
    enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    enc_ff_swish=True,
    enc_ff_glu=True,
    enc_attn_kv_heads=[1, 2, 3, 4][size],
    enc_attn_gate_values=True,
    enc_sandwich_coef=[2, 4, 6, 8][size],
    enc_shift_tokens=1,
    enc_use_abs_pos_emb=False,
    enc_attn_on_attn=True,
    enc_macaron=True,
    enc_resi_dual=True,
    enc_resi_dual_scale=0.1,
    # enc_rotary_pos_emb=True,
    enc_alibi_pos_bias=True,
    enc_alibi_num_heads=[2, 4, 6, 8][size],

    dec_attn_num_mem_kv=[6, 12, 18, 24][size],
    dec_num_memory_tokens=[6, 12, 18, 24][size],
    dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    dec_ff_swish=True,
    dec_ff_glu=True,
    dec_attn_kv_heads=[1, 2, 3, 4][size],
    dec_attn_gate_values=True,
    dec_sandwich_coef=[2, 4, 6, 8][size],
    dec_shift_tokens=1,
    dec_use_abs_pos_emb=False,
    dec_attn_on_attn=True,
    dec_macaron=True,
    dec_resi_dual=True,
    dec_resi_dual_scale=0.1,
    # dec_rotary_pos_emb=True,
    dec_alibi_pos_bias=True,
    dec_alibi_num_heads=[2, 4, 6, 8][size])

  model = model.cuda()
  model = torch.compile(model)

  return model

def tkn(str):
  pass

def tokenize(sp, data: bytes):
  tokens = sp.encode(data)
  return tokens

def detokenize(sp, tokens: [int]):
  tokens = [t for t in tokens if t < NUM_TOKENS-2]
  tokens = sp.decode(tokens)
  return tokens


def gen_yarpgen(threadnum, num):
  ret = []

  yarpgen = f'/{ROOTDIR}/bin/{platform.system()}/yarpgen'
  outdir = f'/{TMP}/yarpgen_{threadnum}'
  os.makedirs(outdir, exist_ok=True)
  c_file = f'{outdir}/func.c'
  opt_obj = f'{outdir}/func.opt.o'
  unopt_obj = f'{outdir}/func.unopt.o'

  for x in range(num):
    if threadnum == 0:
      print(x)

    run(f'{yarpgen} --std=c -o {outdir}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    run(f'clang -c {c_file} -o {unopt_obj} -include stdint.h -O0 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    run(f'clang -c {c_file} -o {opt_obj}   -include stdint.h -O3 -s {CCFLAGS}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    run(f'{OBJCOPY}  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {unopt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    run(f'{OBJCOPY}  --remove-section .eh_frame --remove-section .note.GNU-stack --remove-section .comment --remove-section .llvm_addrsig {opt_obj}'.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    with open(opt_obj, 'rb') as f, open(unopt_obj, 'rb') as g:
      ret.append((f.read(),g.read()))

  return ret


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
