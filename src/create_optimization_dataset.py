import random
import subprocess
import tempfile
import os
import generate_c
import string
from collections import defaultdict

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
    ret_unopt = subprocess.run(f'riscv64-linux-gnu-gcc /tmp/sopt/func{uuid}.c -o /tmp/sopt/func{uuid}_unopt.o -O0 -Wall -c'.split(), capture_output=True)
    ret_opt = subprocess.run(f'riscv64-linux-gnu-gcc /tmp/sopt/func{uuid}.c -o /tmp/sopt/func{uuid}_opt.o -O3 -Wall -c'.split(), capture_output=True)
    if len(ret_opt.stderr) > 0 or len(ret_unopt.stderr) > 0:
      errors =['-Wshift-count-negative','-Woverflow','-Wdiv-by-zero']
      unopt_stderr = ret_unopt.stderr.decode('utf-8')
      opt_stderr = ret_opt.stderr.decode('utf-8')
      for err in errors:
        if err in unopt_stderr or err in opt_stderr:
          print("UB in generated code")
          continue
      break
    else:
      break

  #strip because symbols get in the way of parsing
  stripped_unopt = subprocess.run(f'riscv64-linux-gnu-strip /tmp/sopt/func{uuid}_unopt.o'.split(), capture_output=True)
  stripped_opt = subprocess.run(f'riscv64-linux-gnu-strip /tmp/sopt/func{uuid}_opt.o'.split(), capture_output=True)

  unopt_listing = subprocess.run(f'riscv64-linux-gnu-objdump -M no-aliases --no-show-raw-insn -d /tmp/sopt/func{uuid}_unopt.o'.split(), capture_output=True)
  opt_listing = subprocess.run(f'riscv64-linux-gnu-objdump -M no-aliases --no-show-raw-insn -d /tmp/sopt/func{uuid}_opt.o'.split(), capture_output=True)

  unopt_disasm = unopt_listing.stdout.decode('utf-8')
  opt_disasm = opt_listing.stdout.decode('utf-8')

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
          return None
        if instr in ['lui','c.lui']:
          #convert hex imm to decimal
          imm = asm.split(',')[1]
          newimm = int(imm,16)
          if newimm > 4096:
            #lui can have values > 4096 which we don't support yet so just return None
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

if __name__ == '__main__':

  from riscv_sopt import tokenize_asm, INSTRS, tokenize_prog, detokenize_prog
  INSTRS = set(INSTRS)

  for x in range(100):
    prog = compile(x)
    unopt = []
    if prog is None:
      continue
    unopt_tokenized = tokenize_prog(prog['unopt'], True, 256)
    opt_tokenized   = tokenize_prog(prog['opt'],  False, 256)
    detokenized = detokenize_prog(opt_tokenized)
    if x % 10 == 0:
      for k,v in sorted(USED_INSTRS.items(), key= lambda x: x[1]):
        print(k,v)
      print("unused instrs", sorted(INSTRS - USED_INSTRS.keys()))

    print(prog)
