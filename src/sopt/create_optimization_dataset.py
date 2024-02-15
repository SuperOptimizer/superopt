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
NUM_PROGS = 0
def gen(uuid):
  global NUM_PROGS
  with gzip.open(f'/{ROOTDIR}/data/db_{uuid}.csv.gz','w+t') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writeheader()

    for x in range(10000):
      prog = yarpgen(uuid)
      compiled = compile(prog, CC, STRIP, OBJDUMP)
      if x % 1000 == 0:
        print(uuid,x)
      if compiled is None:
        continue
      unopt = tokenize(compiled['unopt'], True, 256)
      if unopt is None:
        continue
      opt   = tokenize(compiled['opt'],  False, 128)
      if opt is None:
        continue
      NUM_PROGS +=1
      print(NUM_PROGS)
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
        print("already in db")
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
  for uuid in range(16):
    os.makedirs(f'{TMP}/yarpgen_{uuid}', exist_ok=True)
  #with multiprocessing.Pool(16) as p:
  #  p.map(gen,list(range(16)))
  #gen(0)
  ALL_INPUTS = set()
  OUT = list()
  for i,gz in enumerate(os.listdir(f'/{ROOTDIR}/data/')):
    with gzip.open(f'/{ROOTDIR}/data/{gz}','rt') as inf:
      if i == 4:
        print()
      reader = csv.DictReader(inf)
      for row in reader:
        h = hash(row['unopt'])
        if h in ALL_INPUTS:
          continue
        else:
          ALL_INPUTS.add(h)
          OUT.append(row)
  with gzip.open(f'/{ROOTDIR}/data/processed.csv.gz', 'w+t') as outf:
    writer = csv.DictWriter(outf,['c','unopt','opt'])
    writer.writeheader()
    writer.writerows(OUT)

