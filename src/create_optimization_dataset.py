import random
import subprocess
import tempfile

import generate_c

def compile():
  prog = ""
  args = ['a','b','c','d','e','f']
  constants = ['0','1','2','3','4','8','16','32']
  for x in range(10):
    func = generate_c.gen_random_func(random.randint(1,128),
                                      random.sample(args, random.randint(1,len(args))),
                                      random.sample(constants,random.randint(1,len(constants)//2)),
                                      'int',
                                      f'func_{x}')
    #print(func)
    prog += func + '\n'
  return prog

prog = compile()
print(prog)
