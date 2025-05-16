import torch.utils.data
import os
from x_transformers import XTransformer

import torch
from torchao.float8 import convert_to_float8_training
import bitsandbytes as bnb
from torchao.prototype.quantized_training import int8_weight_only_quantized_training, bitnet_training
from torchao import quantize_
MODEL_SIZE = "small"

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
TMP = '/tmp/sopt'

#vocab tokens are the first 0 through NUM_VOCAB_TOKENS-1, used by sentencepiece
NUM_TOKENS = 16384
NUM_SPECIAL_TOKENS = 3
NUM_VOCAB_TOKENS = NUM_TOKENS - NUM_SPECIAL_TOKENS

ENC_SEQ_LEN = int(8192*1)
DEC_SEQ_LEN = int(8192*1)
GENERATE_EVERY = 100
CHECKPOINT_EVERY = 100
LEARNING_RATE = 1e-4
NUM_BATCHES = int(1e7)
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16
DTYPE = torch.bfloat16


def module_filter_fn(mod: torch.nn.Module, fqn: str):
  # don't convert the last module
  if "attn" not in fqn:
    return False
  if "logit" in fqn:
    return False
  if fqn == "1":
    return False
  # don't convert linear modules with weight dimensions not divisible by 16
  if isinstance(mod, torch.nn.Linear):
    if mod.in_features == 3072:
      return True
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
      return False
  return False


def get_model(pad_value):
  size = {'small': 0, 'medium': 1, 'large': 2, 'xl': 3}[MODEL_SIZE]
  model = XTransformer(
    dim=768,
    pad_value=pad_value,
    tie_token_emb=False,
    return_tgt_loss=True,
    ignore_index=pad_value,

    enc_attn_flash=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=4*4,
    enc_heads=4*4,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    #enc_ff_swish=True,
    #enc_ff_glu=True,
    enc_use_abs_pos_emb=False,

    dec_attn_flash=True,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=4*4,
    dec_heads=4*4,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    #dec_ff_swish=True,
    #dec_ff_glu=True,
    dec_use_abs_pos_emb=False,
  )
  model = model.cuda()
  model = model.bfloat16()
  convert_to_float8_training(model, module_filter_fn=module_filter_fn)
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
  elif str == 'EOS':
    return NUM_VOCAB_TOKENS + 2
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


def bytes_to_bitstream(arr: bytes):
  """Convert a bytes array to a string of '0' and '1' characters."""
  bits = ''
  for byte in arr:
    # Convert each byte to its 8-bit binary representation
    bits += format(byte, '08b')
  return bits


def bitstream_to_bytes(bitstream: str):
  """Convert a string of '0' and '1' characters to a bytes array.
     If the length is not a multiple of 8, pad with 0s."""
  # Check if the input is a valid bit string
  if not all(bit in '01' for bit in bitstream):
    return b"invalid"

  # Pad with 0s if necessary to make length a multiple of 8
  padding = 0
  if len(bitstream) % 8 != 0:
    padding = 8 - (len(bitstream) % 8)
    bitstream += '0' * padding

  # Convert each 8-bit group to a byte
  result = bytearray()
  for i in range(0, len(bitstream), 8):
    byte_bits = bitstream[i:i + 8]
    byte_value = int(byte_bits, 2)
    result.append(byte_value)

  return bytes(result)


def tokenize_bytes_to_bitstream(sp, data: bytes):
    """Convert bytes to bitstream, then tokenize."""
    bitstream = bytes_to_bitstream(data)
    tokens = sp.encode(bitstream)
    return tokens

def detokenize_bytes_from_bitstream(sp, tokens: [int]):
    """Detokenize to bitstream, then convert to bytes."""
    if tkn('EOS') in tokens:
        print("got an eos token")
    tokens = [t for t in tokens if t < NUM_VOCAB_TOKENS]
    bitstream = sp.decode(tokens)
    return bitstream_to_bytes(bitstream)

def tokenize_bitstream(sp, bitstream: str):
    """Directly tokenize a bitstream."""
    tokens = sp.encode(bitstream)
    return tokens

def detokenize_bitstream(sp, tokens: [int]):
    """Detokenize to a bitstream."""
    tokens = [t for t in tokens if t < NUM_VOCAB_TOKENS]
    bitstream = sp.decode(tokens)
    return bitstream