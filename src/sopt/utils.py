import random
import string
from types import ModuleType, FunctionType
from gc import get_referents
import sys
from functools import wraps
import time
import os


ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
TMP = '/tmp/sopt'

def randstring(n):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def getsize(obj):
  BLACKLIST = type, ModuleType, FunctionType
  """sum size of object & members."""
  if isinstance(obj, BLACKLIST):
      raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
  seen_ids = set()
  size = 0
  objects = [obj]
  while objects:
    need_referents = []
    for obj in objects:
      if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
        seen_ids.add(id(obj))
        size += sys.getsizeof(obj)
        need_referents.append(obj)
    objects = get_referents(*need_referents)
  return size

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