import torch.utils.data
import os
from x_transformers import XTransformer
import torch

torch._dynamo.config.recompile_limit = 64

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HOMEDIR = os.path.abspath(os.path.expanduser("~"))
TMP = '/tmp/sopt'

#vocab tokens are the first 0 through NUM_VOCAB_TOKENS-1
NUM_VOCAB_TOKENS = 256
NUM_SPECIAL_TOKENS = 3
NUM_TOKENS = NUM_VOCAB_TOKENS + NUM_SPECIAL_TOKENS

LEARNING_RATE = 5e-4
NUM_BATCHES = int(1e7)
GENERATE_EVERY = 1000
CHECKPOINT_EVERY = 100
PRINT_STATS_EVERY = 2

if '4060' in torch.cuda.get_device_name():
  MODEL_SIZE = 'small'
  ENC_SEQ_LEN = DEC_SEQ_LEN = 4096
elif '4090' in torch.cuda.get_device_name():
  MODEL_SIZE = 'medium'
  ENC_SEQ_LEN = DEC_SEQ_LEN = 4096
elif 'H100' in torch.cuda.get_device_name():
  MODEL_SIZE = 'small'
  ENC_SEQ_LEN = DEC_SEQ_LEN = 32768
  BATCH_SIZES = {
    32768: 1,
    16384: 2,
    8192: 7,
    4096: 8*3,
    2048: 8*3*4,
    1024: 8*3*3*4,
  }



def get_model(pad_value):
  size = {'small': 1, 'medium': 2, 'large': 3, 'xl': 4}[MODEL_SIZE]
  model = XTransformer(
    dim=256 * size,
    pad_value=pad_value,
    tie_token_emb=True,
    ignore_index=pad_value,

    enc_attn_flash=True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=4*size,
    enc_heads=4*size,
    enc_max_seq_len=ENC_SEQ_LEN,
    #enc_use_simple_rmsnorm=True,
    enc_ff_no_bias=True,
    #enc_ff_swish=True,
    enc_ff_glu=True,
    #enc_ff_relu_squared = True,
    enc_use_abs_pos_emb=False,
    enc_attn_one_kv_head = True,
    #enc_sandwich_norm = True,
    #enc_layer_dropout = 0.1,
    #enc_attn_dropout = 0.1,
    #enc_ff_dropout = 0.1,
    #enc_macaron=True,
    #enc_shift_tokens = 1,
      #enc_attn_head_scale = True,
      #enc_ff_post_act_ln = True,
      #enc_scale_residual = True,

    dec_attn_flash=True,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=4*size,
    dec_heads=4*size,
    dec_max_seq_len=DEC_SEQ_LEN,
    #dec_use_simple_rmsnorm=True,
    dec_ff_no_bias=True,
    #dec_ff_swish=True,
    dec_ff_glu=True,
    #dec_ff_relu_squared = True,
    dec_use_abs_pos_emb=False,
    dec_attn_one_kv_head = True,
    #dec_sandwich_norm = True,
    #dec_layer_dropout = 0.1,
    #dec_attn_dropout = 0.1,
    #dec_ff_dropout = 0.1,
    #dec_macaron=True,
    #dec_shift_tokens = 1,
      #dec_attn_head_scale = True,
      #dec_ff_post_act_ln = True,
      #dec_scale_residual = True,
  )
  model = model.cuda()
  model = model.bfloat16()
  model = torch.compile(model,dynamic=True) #
  return model

def tkn(str):
  if str == 'PAD':
    return NUM_VOCAB_TOKENS + 0
  elif str == 'DECSTART':
    return NUM_VOCAB_TOKENS + 1
  elif str == 'EOS':
    return NUM_VOCAB_TOKENS + 2
  raise

def bytewise_tokenize(data_bytes):
    return [b for b in data_bytes]  # Each byte becomes a token ID

def bytewise_detokenize(token_ids):
    return bytes(token_ids)