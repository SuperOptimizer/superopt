import random
import subprocess
import tempfile
import os
import generate_c
import string
from collections import defaultdict
from subprocess import Popen, PIPE
import csv
import gzip

from riscv_sopt import tokenize_asm, INSTRS, tokenize_prog, detokenize_prog, tkn

USED_INSTRS = defaultdict(lambda: 0)

def randstring(n):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def compile(args):
  uuid,min_tokens,max_tokens = args
  uuid = randstring(32)
  args = ['a','b','c','d','e','f']
  args = ['*' +a if random.randint(1,2)==1 else '' + a for a in args]
  constants = random.sample(range(-2048,2047),8)
  constants.extend([-16,-8,-4,-2,-1,0,1,2,3,4,5,8,16])
  constants = [str(x) for x in constants]
  while True:
    func = generate_c.gen_random_func(random.randint(min_tokens,max_tokens),
                                        random.sample(args, random.randint(1,len(args))),
                                        random.sample(constants,random.randint(1,len(constants)//4)),
                                        ['char','unsigned char','short','unsigned short','int','unsigned int','long long','unsigned long long'],
                                        f'func_{uuid}')

    with open(f'/tmp/sopt/func{uuid}.c', 'w+') as f:
      f.write(func)

    ret_unopt_process = Popen(
      ['riscv64-linux-gnu-gcc', f'/tmp/sopt/func{uuid}.c', '-o', f'/tmp/sopt/func{uuid}_unopt.o', '-O0',
       '-Wall', '-c'], stdout=PIPE, stderr=PIPE)

    ret_opt_process = Popen(
      ['riscv64-linux-gnu-gcc', f'/tmp/sopt/func{uuid}.c', '-o', f'/tmp/sopt/func{uuid}_opt.o', '-O3',
       '-Wall', '-c'], stdout=PIPE, stderr=PIPE)

    _, unopt_stderr = ret_unopt_process.communicate()
    _, opt_stderr = ret_opt_process.communicate()

    unopt_stderr = unopt_stderr.decode('utf-8')
    opt_stderr = opt_stderr.decode('utf-8')

    if len(opt_stderr) > 0 or len(unopt_stderr) > 0:
      errors =['-Wshift-count-negative','-Woverflow','-Wdiv-by-zero']
      for err in errors:
        if err in unopt_stderr or err in opt_stderr:
          #print("UB in generated code")
          continue
      break
    else:
      break

  # Start the process to strip the unoptimized object file
  stripped_unopt_process = subprocess.Popen(['riscv64-linux-gnu-strip', f'/tmp/sopt/func{uuid}_unopt.o'])
  stripped_opt_process = subprocess.Popen(['riscv64-linux-gnu-strip', f'/tmp/sopt/func{uuid}_opt.o'])
  stripped_unopt_process.wait()
  stripped_opt_process.wait()

  unopt_process = Popen(['riscv64-linux-gnu-objdump', '-M', 'no-aliases', '--no-show-raw-insn', '-d', f'/tmp/sopt/func{uuid}_unopt.o'],stdout=PIPE)
  opt_process = Popen(['riscv64-linux-gnu-objdump', '-M', 'no-aliases', '--no-show-raw-insn', '-d', f'/tmp/sopt/func{uuid}_opt.o'],stdout=PIPE)

  unopt_stdout, unopt_stderr = unopt_process.communicate()
  unopt_disasm = unopt_stdout.decode('utf-8')

  opt_stdout, opt_stderr = opt_process.communicate()
  opt_disasm = opt_stdout.decode('utf-8')

  out = dict()
  out['c'] = func
  for listing in [unopt_disasm, opt_disasm]:
    in_disasm = False
    disasm = []
    for line in listing.split('\n'):
      if not in_disasm:
        if line.strip().startswith('0000000000000000'):
          in_disasm = True
          continue
      else:
        if line.strip() == '':
          break
        if '#' in line:
          line = line.split('#')[0]
        addr, asm = line.strip().split(':')
        addr = addr.strip()
        asm = asm.strip()
        instr = asm.split()[0].strip()
        USED_INSTRS[instr]+=1
        if instr in ['c.ebreak','ebreak']:
          #TODO: ???
          # for now we will just not output the program
          try:
            os.remove(f'/tmp/sopt/func{uuid}_opt.o')
          except:
            pass  # ???
          try:
            os.remove(f'/tmp/sopt/func{uuid}_unopt.o')
          except:
            pass  # ???
          try:
            os.remove(f'/tmp/sopt/func{uuid}.c')
          except:
            pass  # ???
          return None
        if instr in ['lui','c.lui']:
          #convert hex imm to decimal
          imm = asm.split(',')[1]
          newimm = int(imm,16)
          if newimm > 4096:
            #lui can have values > 4096 which we don't support yet so just return None
            try:
              os.remove(f'/tmp/sopt/func{uuid}_opt.o')
            except:
              pass  # ???
            try:
              os.remove(f'/tmp/sopt/func{uuid}_unopt.o')
            except:
              pass  # ???
            try:
              os.remove(f'/tmp/sopt/func{uuid}.c')
            except:
              pass  # ???
            return None
          asm = asm.replace(imm,str(newimm))
        if instr in ['beq','bne','blt','bge','bltu','bgeu','c.beqz','c.bnez', 'c.j']:
          # objdump AFAICT only dumps absolute dump addresses but the instruction is encoded as a relative address
          # so convert to a relative address here
          if instr == 'c.j':
            branchaddr = asm.split()[1]
          else:
            branchaddr = asm.split(',')[-1]
          offset = int(branchaddr,16) - int(addr,16)
          asm = asm.replace(branchaddr, ('+' if offset > 0 else '') + str(offset))
        disasm.append(asm)
    k = 'unopt' if listing == unopt_disasm else 'opt'
    out[k] = '\n'.join(disasm)
  try:
    os.remove(f'/tmp/sopt/func{uuid}_opt.o')
  except:
    pass #???
  try:
    os.remove(f'/tmp/sopt/func{uuid}_unopt.o')
  except:
    pass #???
  try:
    os.remove(f'/tmp/sopt/func{uuid}.c')
  except:
    pass #???
  return out

def gen(uuid):

  with gzip.open(f'/tmp/sopt/db_{uuid}.csv.gz','a+t') as f:
    writer = csv.DictWriter(f,['c','unopt','opt'])
    writer.writeheader()

    for x in range(1000000):
      prog = compile((x,8,32))
      if x % 1000 == 0:
        print(uuid,x)
      if prog is None:
        continue
      unopt = tokenize_prog(prog['unopt'], True, 256)
      opt   = tokenize_prog(prog['opt'],  False, 128)
      if unopt is None or opt is None:
        continue
      try:
        #sometimes 'PAD' doesn't show up in the input
        #I _assume_ this is because we generated exactly 256 tokens
        #but for now let's just skip that case
        row = {'c': prog['c'],
                       'unopt': unopt[:unopt.index(tkn('PAD'))],
                       'opt': opt[:opt.index(tkn('PAD'))]}
      except:
        continue

      writer.writerow(row)

if __name__ == '__main__':
  import multiprocessing

  with multiprocessing.Pool(16) as p:
    p.map(gen,list(range(16)))


  with gzip.open('/tmp/sopt/db_0.csv.gz','rt') as f:
    reader = csv.DictReader(f)
    for row in reader:
      print(row)
