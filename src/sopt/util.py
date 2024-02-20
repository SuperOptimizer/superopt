from functools import wraps
import time
import torch
import random
import string


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
