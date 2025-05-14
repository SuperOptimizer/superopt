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

ARCH = 'x86'
MODEL_SIZE = "small"


DEVICE = 'cuda'

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


def get_model(pad_value):
  size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[MODEL_SIZE]
  model = XTransformer(
    dim=[256, 512, 768, 1024][size],
    pad_value=pad_value,
    tie_token_emb=True,
    return_tgt_loss=True,

    enc_attn_flash=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=4,
    enc_heads=4,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    enc_ff_swish=True,
    enc_ff_glu=True,

    dec_attn_flash=True,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=4,
    dec_heads=4,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    dec_ff_swish=True,
    dec_ff_glu=True,
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
    print("got an invalid hex string in hex_string_to_bytes")
  try:
    if len(hex_string) % 2 == 1:
      return bytes.fromhex(hex_string[:-1])
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

