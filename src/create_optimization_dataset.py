import random
import subprocess
import tempfile
import os
import generate_c

def compile(uuid):
  args = ['a','b','c','d','e','f']
  constants = ['0','1','2','3','4','8','16','32']
  constants = random.sample(range(-2048,2047),8)
  constants = [str(x) for x in constants]
  while True:
    func = generate_c.gen_random_func(random.randint(1,32),
                                        random.sample(args, random.randint(1,len(args))),
                                        random.sample(constants,random.randint(1,len(constants)//2)),
                                        'int',
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

  unopt_listing = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-objdump --no-show-raw-insn -d /tmp/sopt/func{uuid}_unopt.o'.split(), capture_output=True)
  opt_listing = subprocess.run(f'/opt/riscv/bin/riscv64-unknown-elf-objdump --no-show-raw-insn -d /tmp/sopt/func{uuid}_opt.o'.split(), capture_output=True)

  unopt_disasm = unopt_listing.stdout.decode('ascii')
  opt_disasm = opt_listing.stdout.decode('ascii')

  print()

prog = compile(1)
print(prog)
