import iced_x86

#x86 currently uses a character level encoding scheme
#so 256 tokens for byte values, plus 2 for PAD and DECSTART

NUM_TOKENS = 258
METATOKENS = ['PAD','DECSTART','IMMEDIATE_SEPARATOR']

ALL_INSTRS = [
  'adc', 'adcb', 'adcl', 'adcq', 'adcw', 'add', 'addb', 'addl', 'addq', 'addw', 'and', 'andb', 'andl', 'andn', 'andq',
  'andw', 'bt', 'btcq', 'btr', 'bts', 'call', 'cbtw', 'cltd', 'cltq', 'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmove',
  'cmovg', 'cmovge', 'cmovl', 'cmovle', 'cmovne', 'cmovns', 'cmovs', 'cmp', 'cmpb', 'cmpl', 'cmpq', 'cmpw', 'cqto',
  'cwtl', 'dec', 'decb', 'decl', 'decq', 'decw', 'div', 'divl', 'divq', 'divw', 'idiv', 'idivl', 'idivq', 'imul', 'inc',
  'incb', 'incl', 'incq', 'incw', 'ja', 'jae', 'jb', 'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jns', 'js',
  'lea', 'leave', 'mov', 'movabs', 'movb', 'movl', 'movq', 'movsbl', 'movsbq', 'movsbw', 'movslq', 'movswl', 'movswq',
  'movw', 'movzbl', 'movzwl', 'mul', 'mulb', 'neg', 'negb', 'negl', 'nop', 'nopl', 'nopw', 'not', 'notb', 'notl',
  'notw', 'or', 'orb', 'orl', 'orq', 'orw', 'pop', 'push', 'ret', 'ror', 'rorx', 'sar', 'sarx', 'sbb', 'sbbb', 'sbbl',
  'sbbq', 'sbbw', 'seta', 'setae', 'setb', 'setbe', 'sete', 'setg', 'setge', 'setl', 'setle', 'setne', 'shl', 'shlb',
  'shlw', 'shlx', 'shr', 'shrx', 'sub', 'subb', 'subl', 'subq', 'subw', 'test', 'testb', 'testl', 'testq', 'testw',
  'ud2', 'vbroadcastss', 'vextracti128', 'vinserti128', 'vinsertps', 'vmovd', 'vmovddup', 'vmovdqa', 'vmovdqu', 'vmovq',
  'vpackusdw', 'vpackuswb', 'vpaddb', 'vpaddd', 'vpaddq', 'vpaddw', 'vpand', 'vpandn', 'vpblendvb', 'vpbroadcastb',
  'vpbroadcastd', 'vpbroadcastq', 'vpbroadcastw', 'vpcmpeqb', 'vpcmpeqd', 'vpcmpeqq', 'vpcmpeqw', 'vpcmpgtd',
  'vpcmpgtq', 'vperm2i128', 'vpermq', 'vpextrb', 'vpextrd', 'vpextrq', 'vpextrw', 'vphminposuw', 'vpinsrb', 'vpinsrd',
  'vpinsrw', 'vpmaskmovq', 'vpmaxsb', 'vpmaxsd', 'vpmaxsw', 'vpmaxud', 'vpmaxuw', 'vpminsb', 'vpminsd', 'vpminsw',
  'vpminub', 'vpminud', 'vpminuw', 'vpmovsxbw', 'vpmovsxdq', 'vpmovsxwd', 'vpmovzxbw', 'vpmovzxdq', 'vpmovzxwd',
  'vpmulld', 'vpmullw', 'vpmuludq', 'vpor', 'vpsadbw', 'vpshufb', 'vpshufd', 'vpshuflw', 'vpslld', 'vpsllq', 'vpsrld',
  'vpsrldq', 'vpsrlq', 'vpsubb', 'vpsubd', 'vpsubq', 'vpunpckhbw', 'vpunpcklbw', 'vpunpckldq', 'vpunpcklqdq',
  'vpunpcklwd', 'vpxor', 'vshufps', 'vzeroupper', 'xchg', 'xor', 'xorb', 'xorl', 'xorq', 'xorw']

ALL_REGS = [
  '%ah', '%al', '%ax', '%bl', '%bp', '%bpl', '%bx', '%cl', '%cx', '%di', '%dil', '%dl', '%dx', '%eax', '%ebp', '%ebx',
  '%ecx', '%edi', '%edx', '%esi', '%r10', '%r10b', '%r10d', '%r10w', '%r11', '%r11b', '%r11d', '%r11w', '%r12', '%r12b',
  '%r12d', '%r12w', '%r13', '%r13b', '%r13d', '%r13w', '%r14', '%r14b', '%r14d', '%r14w', '%r15', '%r15b', '%r15d',
  '%r15w', '%r8', '%r8b', '%r8d', '%r8w', '%r9', '%r9b', '%r9d', '%r9w', '%rax', '%rbp', '%rbx', '%rcx', '%rdi', '%rdx',
  '%rip', '%rsi', '%rsp', '%si', '%sil', '%xmm0', '%xmm1', '%xmm10', '%xmm11', '%xmm12', '%xmm13', '%xmm14', '%xmm15',
  '%xmm2', '%xmm3', '%xmm4', '%xmm5', '%xmm6', '%xmm7', '%xmm8', '%xmm9', '%ymm0', '%ymm1', '%ymm10', '%ymm11',
  '%ymm12', '%ymm13', '%ymm14', '%ymm15', '%ymm2', '%ymm3', '%ymm4', '%ymm5', '%ymm6', '%ymm7', '%ymm8', '%ymm9']


ALL_ARGS = set()

def tkn(t: str):
  if t.startswith('$'):
    #idk what this means but we don't need it AFAICT
    t = t[1:]
  if t.startswith('0x') or t.startswith('-0x'):
    val = int(t, 16)
    if val >= 0:
      if val <= 127:
        return val
      elif val <= 32767:
        return [(val >> 8) & 0xff, val & 0xff]
      elif val <= 2147483647:
        return [(val >> 24) & 0xff, (val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff]
      elif val <= 9223372036854775807:
        return [(val >> 56) & 0xff, (val >> 48) & 0xff, (val >> 40) & 0xff, (val >> 32) & 0xff,
               (val >> 24) * 0xff, (val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff]
    else:
      if val >= -128:
        return 256 + val
      elif val >= -32768:
        val += 65536
        return [val >> 8 & 0xff, val & 0xff]
      elif val >= -2147483648:
        val += 4294967296
        return [(val >> 24) & 0xff, (val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff]
      elif val >= 9223372036854775808:
        val += 18446744073709551616
        return [(val >> 56) & 0xff, (val >> 48) & 0xff, (val >> 40) & 0xff, (val >> 32) & 0xff,
                (val >> 24) * 0xff, (val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff]


  #elif t.startswith()
  elif t in METATOKENS:
    return 256 + METATOKENS.index(t)
  else:
    return int(t, 10)

def tokenize(prog: str, encoder: bool, ctxlen: int):
  '''prog is the output from objdump'''

  if '.text.unlikely' in prog:
    return None

  ret = []
  in_disasm = False
  for line in prog.split('\n'):
    if line == '':
      continue
    if line.startswith('00000'):
      in_disasm = True
      continue
    if in_disasm:
      print(line)
      if '#' in line:
        line = line.split('#')[0]
      line = line.strip()
      pc, bytes, asm = line.split('\t')
      try:
        instr,args = asm.split()

      except:
        instr = asm
        args = ''
      instr = instr.strip()
      args = args.strip()
      args = args.replace('(',' ').replace(')',' ').replace(',',' ')
      for a in args.split():
        if a.startswith('$'):
          ALL_ARGS.add(a)
  print(sorted(ALL_ARGS))
  print(len(ALL_ARGS))
  return True

def detokenize_char(prog:[int]):
  ret = []
  prog = [x for x in prog if x < 256]
  decoder = iced_x86.Decoder(64, bytes(prog), 0)

  formatter = iced_x86.Formatter(iced_x86.FormatterSyntax.GAS)
  for instr in decoder:
    disasm = formatter.format(instr)
    start_index = instr.ip - 0
    bytes_str = bytes(prog)[start_index:start_index + instr.len].hex().lower()
    ret.append(f"\t{instr.ip:0X}\t{bytes_str}\t{disasm}")
  return '\n'.join(ret)


def tokenize_char(prog: str, encoder: bool, ctxlen: int):
  '''character level tokenizer'''
  if 'text.unlikely' in prog:
    return None
  ret = []
  if not encoder:
    ret.append(tkn('DECSTART'))
  in_disasm = False
  for line in prog.split('\n'):
    if line == '':
      continue
    if line.startswith('00000'):
      in_disasm = True
      continue
    if in_disasm:
      try:
        pc,bytes,asm = line.split('\t')
      except:
        pc,bytes = line.split('\t')
      for byte in bytes.split():
        ret.append(int(byte,16))
  if len(ret) > ctxlen:
    return None
  for x in range(ctxlen - len(ret)):
    ret.append(tkn('PAD'))
  return ret
