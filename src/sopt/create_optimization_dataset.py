import os
import csv
import gzip
import platform
import multiprocessing

from riscv import tokenize, tkn
from utils import  ROOTDIR, TMP
from gen import compile, yarpgen



ARCH = 'riscv'

if platform.system() == 'Linux':
  if ARCH == 'riscv':
    CC = 'riscv64-linux-gnu-gcc'
    STRIP = 'riscv64-linux-gnu-strip'
    OBJDUMP = 'riscv64-linux-gnu-objdump'
  elif ARCH == 'x86':
    CC = 'gcc'
    STRIP = 'strip'
    OBJDUMP = 'objdump'



ALL_INPUTS = set()
def gen(uuid):
  with open(f'/{TMP}/data/db_{uuid}.csv.gz','w+t') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writeheader()

    for x in range(1000):
      if uuid == 0 and x % 10 == 0:
          print(x)
      prog = yarpgen(uuid)
      compiled = compile(prog, CC, STRIP, OBJDUMP)
      if compiled is None:
        continue
      unopt = tokenize(compiled['unopt'], True, 768)
      if unopt is None:
        continue
      opt   = tokenize(compiled['opt'],  False, 256)
      if opt is None:
        continue
      #sometimes 'PAD' doesn't show up in the input
      #I _assume_ this is because we generated exactly 256 tokens
      if tkn('PAD') in unopt:
        unopt_val = unopt[:unopt.index(tkn('PAD'))]
      else:
        unopt_val = unopt
      if tkn('PAD') in opt:
        opt_val = opt[:opt.index(tkn('PAD'))]
      else:
        opt_val = opt
      if hash(str(unopt_val)) in ALL_INPUTS:
        #print("already in db")
        #this won't eliminate duplicates across processes but will in theory cap the number of duplicates
        #of any given program to num processes
        continue
      else:
        ALL_INPUTS.add(hash(str(unopt_val)))
      row = {'c': compiled['c'],
                     'unopt': unopt_val,
                     'opt': opt_val}

      writer.writerow(row)

if __name__ == '__main__':
  ncpu = multiprocessing.cpu_count()
  print(f"spawning {ncpu} threads")
  ALL_INPUTS = set()
  for x in range(100):
    print('processed',x)
    for uuid in range(ncpu):
      os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
      os.makedirs(f'{TMP}/data', exist_ok=True)
    with multiprocessing.Pool(ncpu) as p:
      p.map(gen,list(range(ncpu)))
    #gen(0)
    OUT = list()
    for i,gz in enumerate(os.listdir(f'/{TMP}/data/')):
      with open(f'/{TMP}/data/{gz}','rt') as inf:
        reader = csv.DictReader(inf)
        for row in reader:
          h = hash(row['unopt'])
          if h in ALL_INPUTS:
            continue
          else:
            ALL_INPUTS.add(h)
            OUT.append(row)
    with gzip.open(f'/{ROOTDIR}/data/processed_{x}.csv.gz', 'w+t') as outf:
      writer = csv.DictWriter(outf,['c','unopt','opt'])
      writer.writeheader()
      writer.writerows(OUT)

