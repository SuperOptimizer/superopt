import random
import subprocess
import tempfile
import os
import generate_c
from collections import defaultdict

USED_INSTRS = defaultdict(lambda: 0)

def compile(uuid):
  args = ['a','b','c','d','e','f']
  args = ['*' +a if random.randint(1,2)==1 else '' + a for a in args]
  constants = random.sample(range(-2048,2047),8)
  constants.extend([-16,-8,-4,-2,-1,0,1,2,3,4,5,8,16])
  constants = [str(x) for x in constants]
  while True:
    func = generate_c.gen_random_func(random.randint(1,24),
                                        random.sample(args, random.randint(1,len(args))),
                                        random.sample(constants,random.randint(1,len(constants)//4)),
                                        ['char','unsigned char','short','unsigned short','int','unsigned int','long long','unsigned long long'],
                                        f'func_{uuid}')
    #func = 'int func(int a, int b, int c){return 1 >> 100;}'
    with open(f'/tmp/sopt/func{uuid}.c', 'w+') as f:
      f.write(func)
    ret_unopt = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-gcc /tmp/sopt/func{uuid}.c -o /tmp/sopt/func{uuid}_unopt.o -O0 -Wall -c'.split(), capture_output=True)
    ret_opt = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-gcc /tmp/sopt/func{uuid}.c -o /tmp/sopt/func{uuid}_opt.o -O3 -Wall -c'.split(), capture_output=True)
    if len(ret_opt.stderr) > 0 or len(ret_unopt.stderr) > 0:
      #generated code had UB so nothing
      continue
    else:
      break

  #strip because symbols get in the way of parsing
  stripped_unopt = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-strip /tmp/sopt/func{uuid}_unopt.o'.split(), capture_output=True)
  stripped_opt = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-strip /tmp/sopt/func{uuid}_opt.o'.split(), capture_output=True)

  unopt_listing = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-objdump -M no-aliases --no-show-raw-insn -d /tmp/sopt/func{uuid}_unopt.o'.split(), capture_output=True)
  opt_listing = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-objdump -M no-aliases --no-show-raw-insn -d /tmp/sopt/func{uuid}_opt.o'.split(), capture_output=True)

  unopt_disasm = unopt_listing.stdout.decode('ascii')
  opt_disasm = opt_listing.stdout.decode('ascii')

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
  return out


from riscv_sopt import tokenize_asm, INSTRS
INSTRS = set(INSTRS)

for x in range(1000):
  prog = compile(x)
  unopt = []
  if prog is None:
    continue
  for line in prog['unopt'].split('\n'):
    tokenized = tokenize_asm(line.strip())
  if x % 10 == 0:
    for k,v in sorted(USED_INSTRS.items(),key= lambda x: x[1]):
      print(k,v)
    print("unused instrs", sorted(INSTRS - USED_INSTRS.keys()))

  print(prog)
