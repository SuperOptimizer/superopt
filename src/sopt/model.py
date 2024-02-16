from x_transformers import XTransformer
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np

ENC_SEQ_LEN = 768
DEC_SEQ_LEN = 256


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
    dim = 256
    batch_size = 2
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
    resi_dual_scale=0.1,)

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
  print(f"num params {params//1024//1024}M {params//1024}K ")

  return model, dtype, ENC_SEQ_LEN, DEC_SEQ_LEN, batch_size, generate_every
