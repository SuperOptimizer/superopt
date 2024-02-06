import parse
import random
# riscv transformer optimizer RISCVSopt
# RISCVSopt is a transformer based optimzier for riscv. riscv instructions are translated into a series of tokens
# and riscvsopt outputs an optimized sequence of instructions
#
# RISCV instructions are mapped to a token vocabulary and sequence of tokens as follows
#
# GPRs are encoded as series zero,ra,sp,gp,tp,t0,t1,t2,s0,s1,a0-a7,s2-s11,t3,t4,t5,t6
# FPRs are encoded as series ft0-ft7, fs0,fs1,fa0-fa7,fs2-fs11,ft8-ft11
# VPRs are encoded as series v0 - v31
#
# TKNs
# + 4096 = immediate values
# + 32   = GPR dest
# + 32   = FPR dest
# + 32   = VPR dest
# + 32   = GPR source
# + 32   = FPR source
# + 32   = VPR source
# + 1024 = instructions
# + *    = meta tokens
#
#  riscv instructions are translated into a sequence of 1 - 5 tokens
# <instr>
# <instr> <imm>
# <instr> <rd>  <imm>
# <instr> <rd>  <rs>
# <instr> <rd>  <rs1> <rs2>
# <instr> <rd>  <rs1> <shamt>
# <instr> <rd>  <rs1> <imm>
# <instr> <rd>  <imm> <rs1>
# <instr> <rs1> <rs2> <imm> # store types
# <instr> <rd>  <imm[19:8]> <imm[7:0]> # lui and auipc
# <instr> <rd>  <rs1> <rs2> <rs3>


GPRS = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2', 's0', 's1', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
        'a7', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']
FPRS = ['ft0', 'ft1', 'ft2', 'ft3', 'ft4', 'ft5', 'ft6', 'ft7', 'fs0', 'fs1', 'fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5',
        'fa6', 'fa7', 'fs2', 'fs3', 'fs4', 'fs5', 'fs6', 'fs7', 'fs8', 'fs9', 'fs10', 'fs11', 'ft8', 'ft9', 'ft10',
        'ft11']
VPRS = ['v' + str(x) for x in range(32)]
INSTRS = ['lui', 'auipc', 'addi', 'slti', 'sltiu', 'xori', 'ori', 'andi', 'slli', 'srli', 'srai', 'add', 'sub', 'sll',
          'slt', 'sltu', 'xor', 'srl', 'sra', 'or', 'and', 'fence', 'fence.i', 'csrrw', 'csrrs', 'csrrc', 'csrrwi',
          'csrrsi', 'csrrci', 'ecall', 'ebreak', 'uret', 'sret', 'mret', 'wfi', 'sfence.vma', 'lb', 'lh', 'lw', 'lbu',
          'lhu', 'sb', 'sh', 'sw', 'jal', 'jalr', 'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'addiw', 'slliw', 'srliw',
          'sraiw', 'addw', 'subw', 'sllw', 'srlw', 'sraw', 'lwu', 'ld', 'sd', 'mul', 'mulh', 'mulhsu', 'mulhu', 'div',
          'divu', 'rem', 'remu', 'mulw', 'divw', 'divuw', 'remw', 'remuw', 'lr.w', 'sc.w', 'amoswap.w', 'amoadd.w',
          'amoxor.w', 'amoand.w', 'amoor.w', 'amomin.w', 'amomax.w', 'amominu.w', 'amomaxu.w', 'lr.d', 'sc.d',
          'amoswap.d', 'amoadd.d', 'amoxor.d', 'amoand.d', 'amoor.d', 'amomin.d', 'amomax.d', 'amominu.d',
          'amomaxu.d', 'fmadd.s', 'fmsub.s', 'fnmsub.s', 'fnmadd.s', 'fadd.s', 'fsub.s', 'fmul.s', 'fdiv.s', 'fsqrt.s',
          'fsgnj.s', 'fsgnjn.s', 'fsgnjx.s', 'fmin.s', 'fmax.s', 'fcvt.w.s', 'fcvt.wu.s', 'fmc.x.w', 'feq.s', 'flt.s',
          'fle.s', 'fclass.s', 'fcvt.s.w', 'fcvt.s.wu', 'fmv.w.x', 'fmadd.d', 'fmsub.d', 'fnmsub.d', 'fnmadd.d',
          'fadd.d', 'fsub.d', 'fmul.d', 'fdiv.d', 'fsqrt.d', 'fsgnj.d', 'fsgnjn.d', 'fsgnjx.d', 'fmin.d', 'fmax.d',
          'fcvt.s.d', 'fcvt.d.s', 'feq.d', 'flt.d', 'fle.d', 'fclass.d', 'fcvt.w.d', 'fcvt.wu.d', 'fcvt.d.w',
          'fcvt.d.wu', 'flw', 'fsw', 'fld', 'fsd', 'fcvt.l.s', 'fcvt.lu.s', 'fcvt.s.l', 'fcvt.s.lu', 'fcvt.l.d',
          'fcvt.lu.d', 'fmv.x.d', 'fcvt.d.l', 'fcvt.d.lu', 'fmv.d.x', 'c.addi4spn', 'c.fld', 'c.lw', 'c.flw', 'c.ld',
          'c.fsd', 'c.sw', 'c.fsw', 'c.sd', 'c.nop', 'c.addi', 'c.jal', 'c.addiw', 'c.li', 'c.addi16sp', 'c.lui',
          'c.srli', 'c.srai', 'c.andi', 'c.sub', 'c.xor', 'c.or', 'c.and', 'c.subw', 'c.addw', 'c.j', 'c.beqz',
          'c.bnez', 'c.slli', 'c.fldsp', 'c.lwsp', 'c.flwsp', 'c.ldsp', 'c.jr', 'c.mv', 'c.ebreak', 'c.jalr', 'c.add',
          'c.fsdsp', 'c.swsp', 'c.fswsp', 'c.sdsp'
          ]
METAS = ['PAD', 'ENCSTART', 'ENCEND', 'DECSTART', 'DECEND']


IMM_TKN_OFF     = 0
GPRDEST_TKN_OFF = IMM_TKN_OFF + 4096
FPRDEST_TKN_OFF = GPRDEST_TKN_OFF + 32
VPRDEST_TKN_OFF = FPRDEST_TKN_OFF + 32
GPRSRC_TKN_OFF  = VPRDEST_TKN_OFF + 32
FPRSRC_TKN_OFF  = GPRSRC_TKN_OFF + 32
VPRSRC_TKN_OFF  = FPRSRC_TKN_OFF + 32
INSTR_TKN_OFF   = VPRSRC_TKN_OFF + 32
META_TKN_OFF    = INSTR_TKN_OFF + 1024


NUM_TOKENS = META_TKN_OFF + len(METAS)

formats = {
  "lr.d": "rd,rs1",
  "sc.d": "rd,rs1,rs2",
  "amoswap.d": "rd,rs2,(rs1)",
  "amoadd.d": "rd,rs2,(rs1)",
  "amoxor.d": "rd,rs2,(rs1)",
  "amoand.d": "rd,rs2,(rs1)",
  "amoor.d": "rd,rs2,(rs1)",
  "amomin.d": "rd,rs2,(rs1)",
  "amomax.d": "rd,rs2,(rs1)",
  "amominu.d": "rd,rs2,(rs1)",
  "amomaxu.d": "rd,rs2,(rs1)",
  "fcvt.l.d": "rd,rs1",
  "fcvt.lu.d": "rd,rs1",
  "fmv.x.d": "rd,rs1",
  "fcvt.d.l": "rd,rs1",
  "fcvt.d.lu": "rd,rs1",
  "fmv.d.x": "rd,rs1",
  "fcvt.l.s": "rd,rs1",
  "fcvt.lu.s": "rd,rs1",
  "fcvt.s.l": "rd,rs1",
  "fcvt.s.lu": "rd,rs1",
  "addiw": "rd,rs1,imm",
  "slliw": "rd,rs1,shamt",
  "srliw": "rd,rs1,shamt",
  "sraiw": "rd,rs1,shamt",
  "addw": "rd,rs1,rs2",
  "subw": "rd,rs1,rs2",
  "sllw": "rd,rs1,rs2",
  "srlw": "rd,rs1,rs2",
  "sraw": "rd,rs1,rs2",
  "lwu": "rd,offset(rs1)",
  "ld": "rd,offset(rs1)",
  "sd": "rs2,offset(rs1)",
  "mulw": "rd,rs1,rs2",
  "divw": "rd,rs1,rs2",
  "divuw": "rd,rs1,rs2",
  "remw": "rd,rs1,rs2",
  "remuw": "rd,rs1,rs2",
  "lr.w": "rd,rs1",
  "sc.w": "rd,rs1,rs2",
  "amoswap.w": "rd,rs2,(rs1)",
  "amoadd.w": "rd,rs2,(rs1)",
  "amoxor.w": "rd,rs2,(rs1)",
  "amoand.w": "rd,rs2,(rs1)",
  "amoor.w": "rd,rs2,(rs1)",
  "amomin.w": "rd,rs2,(rs1)",
  "amomax.w": "rd,rs2,(rs1)",
  "amominu.w": "rd,rs2,(rs1)",
  "amomaxu.w": "rd,rs2,(rs1)",
  "c.addi4spn": "rd,sp,uimm",
  "c.fld": "rd,uimm(rs1)",
  "c.lw": "rd,uimm(rs1)",
  "c.flw": "rd,uimm(rs1)",
  "c.ld": "rd,uimm(rs1)",
  "c.fsd": "rd,uimm(rs1)",
  "c.sw": "rd,uimm(rs1)",
  "c.fsw": "rd,uimm(rs1)",
  "c.sd": "rd,uimm(rs1)",
  "c.nop": "",
  # "c.addi": "rd,u[12:12]|u[6:2]",
  "c.addi": "rd,imm",
  "c.jal": "offset",
  "c.addiw": "rd,imm",
  "c.li": "rd,imm",
  "c.addi16sp": "sp,imm",
  "c.lui": "rd,imm",
  "c.srli": "rd,uimm",
  "c.srai": "rd,uimm",
  "c.andi": "rd,imm",
  "c.sub": "rd,rs2",
  "c.xor": "rd,rs2",
  "c.or": "rd,rs2",
  "c.and": "rd,rs2",
  "c.subw": "rd,rs2",
  "c.addw": "rd,rs2",
  "c.j": "offset",
  "c.beqz": "rs1,offset",
  "c.bnez": "rs1,offset",
  "c.slli": "rd,uimm",
  "c.fldsp": "rd,uimm(sp)",
  "c.lwsp": "rd,uimm(sp)",
  "c.flwsp": "rd,uimm(sp)",
  "c.ldsp": "rd,uimm(sp)",
  "c.jr": "rs1",
  "c.mv": "rd,rs2",
  "c.ebreak": "",
  "c.jalr": "rd",
  "c.add": "rd,rs2",
  "c.fsdsp": "rs2,uimm(sp)",
  "c.swsp": "rs2,uimm(sp)",
  "c.fswsp": "rs2,uimm(rs2)",
  "c.sdsp": "rs2,uimm(sp)",
  "fmadd.s": "rd,rs1,rs2,rs3",
  "fmsub.s": "rd,rs1,rs2,rs3",
  "fnmsub.s": "rd,rs1,rs2,rs3",
  "fnmadd.s": "rd,rs1,rs2,rs3",
  "fadd.s": "rd,rs1,rs2",
  "fsub.s": "rd,rs1,rs2",
  "fmul.s": "rd,rs1,rs2",
  "fdiv.s": "rd,rs1,rs2",
  "fsqrt.s": "rd,rs1",
  "fsgnj.s": "rd,rs1,rs2",
  "fsgnjn.s": "rd,rs1,rs2",
  "fsgnjx.s": "rd,rs1,rs2",
  "fmin.s": "rd,rs1,rs2",
  "fmax.s": "rd,rs1,rs2",
  "fcvt.w.s": "rd,rs1",
  "fcvt.wu.s": "rd,rs1",
  "fmv.x.w": "rd,rs1",
  "feq.s": "rd,rs1,rs2",
  "flt.s": "rd,rs1,rs2",
  "fle.s": "rd,rs1,rs2",
  "fclass.s": "rd,rs1",
  "fcvt.s.w": "rd,rs1",
  "fcvt.s.wu": "rd,rs1",
  "fmv.w.x": "rd,rs1",
  "fmadd.d": "rd,rs1,rs2,rs3",
  "fmsub.d": "rd,rs1,rs2,rs3",
  "fnmsub.d": "rd,rs1,rs2,rs3",
  "fnmadd.d": "rd,rs1,rs2,rs3",
  "fadd.d": "rd,rs1,rs2",
  "fsub.d": "rd,rs1,rs2",
  "fmul.d": "rd,rs1,rs2",
  "fdiv.d": "rd,rs1,rs2",
  "fsqrt.d": "rd,rs1",
  "fsgnj.d": "rd,rs1,rs2",
  "fsgnjn.d": "rd,rs1,rs2",
  "fsgnjx.d": "rd,rs1,rs2",
  "fmin.d": "rd,rs1,rs2",
  "fmax.d": "rd,rs1,rs2",
  "fcvt.s.d": "rd,rs1",
  "fcvt.d.s": "rd,rs1",
  "feq.d": "rd,rs1,rs2",
  "flt.d": "rd,rs1,rs2",
  "fle.d": "rd,rs1,rs2",
  "fclass.d": "rd,rs1",
  "fcvt.w.d": "rd,rs1",
  "fcvt.wu.d": "rd,rs1",
  "fcvt.d.w": "rd,rs1",
  "fcvt.d.wu": "rd,rs1",
  "flw": "rd,offset(rs1)",
  "fsw": "rs2,offset(rs1)",
  "fld": "rd,rs1,offset",
  "fsd": "rs2,offset(rs1)",
  "lui": "rd,imm",
  "auipc": "rd,imm",
  "addi": "rd,rs1,imm",
  "slti": "rd,rs1,imm",
  "sltiu": "rd,rs1,imm",
  "xori": "rd,rs1,imm",
  "ori": "rd,rs1,imm",
  "andi": "rd,rs1,imm",
  "slli": "rd,rs1,shamt",
  "srli": "rd,rs1,shamt",
  "srai": "rd,rs1,shamt",
  "add": "rd,rs1,rs2",
  "sub": "rd,rs1,rs2",
  "sll": "rd,rs1,rs2",
  "slt": "rd,rs1,rs2",
  "sltu": "rd,rs1,rs2",
  "xor": "rd,rs1,rs2",
  "srl": "rd,rs1,rs2",
  "sra": "rd,rs1,rs2",
  "or": "rd,rs1,rs2",
  "and": "rd,rs1,rs2",
  "fence": "",
  "fence.i": "",
  "csrrw": "rd,offset,rs1",
  "csrrs": "rd,offset,rs1",
  "csrrc": "rd,offset,rs1",
  "csrrwi": "rd,offset,uimm",
  "csrrsi": "rd,offset,uimm",
  "csrrci": "rd,offset,uimm",
  "ecall": "",
  "ebreak": "",
  "uret": "",
  "sret": "",
  "mret": "",
  "wfi": "",
  "sfence.vma": "rs1,rs2",
  "lb": "rd,offset(rs1)",
  "lh": "rd,offset(rs1)",
  "lw": "rd,offset(rs1)",
  "lbu": "rd,offset(rs1)",
  "lhu": "rd,offset(rs1)",
  "sb": "rs2,offset(rs1)",
  "sh": "rs2,offset(rs1)",
  "sw": "rs2,offset(rs1)",
  "jal": "rd,offset",
  "jalr": "rd,rs1,offset",
  "beq": "rs1,rs2,offset",
  "bne": "rs1,rs2,offset",
  "blt": "rs1,rs2,offset",
  "bge": "rs1,rs2,offset",
  "bltu": "rs1,rs2,offset",
  "bgeu": "rs1,rs2,offset",
  "mul": "rd,rs1,rs2",
  "mulh": "rd,rs1,rs2",
  "mulhsu": "rd,rs1,rs2",
  "mulhu": "rd,rs1,rs2",
  "div": "rd,rs1,rs2",
  "divu": "rd,rs1,rs2",
  "rem": "rd,rs1,rs2",
  "remu": "rd,rs1,rs2",
}

def detkn(t: int):
  if IMM_TKN_OFF <= t < GPRDEST_TKN_OFF:
    return str(t)
  elif t < FPRDEST_TKN_OFF:
    return GPRS[t - GPRDEST_TKN_OFF]
  elif t < VPRDEST_TKN_OFF:
    return FPRS[t - FPRDEST_TKN_OFF]
  elif t < GPRSRC_TKN_OFF:
    return VPRS[t - VPRDEST_TKN_OFF]
  elif t < FPRSRC_TKN_OFF:
    return GPRS[t - GPRSRC_TKN_OFF]
  elif t < VPRSRC_TKN_OFF:
    return FPRS[t - FPRSRC_TKN_OFF]
  elif t < INSTR_TKN_OFF:
    return VPRS[t - VPRSRC_TKN_OFF]
  elif t < META_TKN_OFF:
    try:
      return INSTRS[t - INSTR_TKN_OFF]
    except:
      return f"invalid_instr{t}"
  else:
    return METAS[t - META_TKN_OFF]

def tkn(t: str, dest=None):
  try:
    val = int(t, 16 if t.startswith('0x') else 10)
    if val < 0:
      val = 4096 + val
    return IMM_TKN_OFF + val
  except:
    pass
  if t in METAS:
    return META_TKN_OFF + METAS.index(t)
  assert dest is not None
  if dest:
    if t in GPRS:
      return GPRDEST_TKN_OFF + GPRS.index(t)
    elif t in FPRS:
      return FPRDEST_TKN_OFF + FPRS.index(t)
    elif t in VPRS:
      return VPRDEST_TKN_OFF + VPRS.index(t)
    else:
      assert False
  else:
    if t in GPRS:
      return GPRSRC_TKN_OFF + GPRS.index(t)
    elif t in FPRS:
      return FPRSRC_TKN_OFF + FPRS.index(t)
    elif t in VPRS:
      return VPRSRC_TKN_OFF + VPRS.index(t)
    else:
      assert False


def tokenize_prog(prog: str, encoder, ctxlen):
  '''tokenize a program for input/output into model'''
  ret = [] if encoder else [tkn('DECSTART')]
  #ret = [tkn('ENCSTART' if encoder else 'DECSTART')]
  for line in prog.split('\n'):
    ret.extend(tokenize_asm(line.strip()))
  #ret.append(tkn('ENCEND' if encoder else 'DECEND'))
  if ctxlen < len(ret):
    print(f"got {len(ret)} tokens but only {ctxlen} context length")
    return None
  else:
    pass
    #print(f"{'encoder' if encoder else 'decoder'} got {len(ret)} tokens ")
  for x in range(ctxlen - len(ret)):
    ret.append(tkn('PAD'))

  return ret

def detokenize_prog(prog: [int]):
  ret = []
  cur = []
  for tkn in prog:
    s = detkn(tkn)
    if s in INSTRS:
      if len(cur) > 0:
        fmt_str = "{instr}\t" + get_fmt_str(cur[0])
        #fmt_str has named arguments, but cur is positional without names
        #so just replace all the named things with {}
        fmt_str = (fmt_str.replace('{instr}','{}')
                   .replace('{rd}','{}')
                   .replace('{rs1}','{}')
                   .replace('{rs2}','{}')
                   .replace('{rs3}','{}')
                   .replace('{shamt}','{}')
                   .replace('{uimm}','{}')
                   .replace('{imm}','{}')
                   .replace('{offset}','{}'))
        try:
          ret.append(fmt_str.format(*cur))
        except:
          ret.append(f"invalid{cur}")
        cur = [s]
        continue
      else:
        cur.append(s)
    elif s in METAS:
      #don't need meta tokens in asm output
      continue
    else:
      cur.append(s)
  #cur[0] _should_ be an instruction but if the network is bad then it may generate illegal code and cur[0] won't be a valid instruction
  try:
    fmt_str = "{instr}\t" + get_fmt_str(cur[0])
  except:
    fmt_str = f"invalid{cur}"
  #fmt_str has named arguments, but cur is positional without names
  #so just replace all the named things with {}
  fmt_str = (fmt_str.replace('{instr}','{}')
             .replace('{rd}','{}')
             .replace('{rs1}','{}')
             .replace('{rs2}','{}')
             .replace('{rs3}','{}')
             .replace('{shamt}','{}')
             .replace('{uimm}','{}')
             .replace('{imm}','{}')
             .replace('{offset}','{}'))
  try:
    ret.append(fmt_str.format(*cur))
  except:
    ret.append(f"invalid {cur}")
  return '\n'.join(ret)

def get_fmt_str(instr):
  try:
    args_fmt = formats[instr]
  except:
    return f'invalid{instr}'
  args_fmt = (args_fmt.replace('rs1', '{rs1}')
              .replace('rs2', '{rs2}')
              .replace('rs3', '{rs3}')
              .replace('offset', '{offset}')
              .replace('shamt', '{shamt}'))
  if 'uimm' in args_fmt:
    args_fmt = args_fmt.replace('uimm', '{uimm}')
  else:
    args_fmt = args_fmt.replace('imm', '{imm}')
  args_fmt = args_fmt.replace('rd\'', 'rd')
  args_fmt = args_fmt.replace('rd', '{rd}')
  return args_fmt

def constprop_gen():
  #generate data to teach constant propagation
  #e.g.
  # c.addiw a0,1
  # c.addiw a0,1
  # c.jr ra
  # to
  # c.addiw a0,2
  # c.jr, ra

  constants = []
  for x in range(random.randint(2,15)):
    constants.append(random.randint(0,(2047-sum(constants))//2))

  rd = random.choice(GPRS)
  stmts = []
  for constant in constants:
    if 0 < constant <= 31:
      stmts.append(f'c.addiw\t{rd},{constant}')
    elif 32 <= constant <= 2047:
      stmts.append(f'addiw\t{rd},{rd},{constant}')
  random.shuffle(stmts)
  out = dict()
  out['unopt'] = '\n'.join(stmts)
  out['opt'] = f'addiw\t{rd},{rd},{sum(constants)}'
  return out


def tokenize_asm(asm: str):
  # str should be assembly string from say objdump -d with -M no-aliases
  # constants are in base 10
  # todo: support pseudo instructions
  instr,args = asm.split()
  args_fmt = get_fmt_str(instr)
  parsed = parse.parse(args_fmt, args)
  ret = [INSTR_TKN_OFF + INSTRS.index(instr)]
  for k, v in parsed.named.items():
    if k == 'rd':
      t = tkn(v, True)
      ret.append(t)
    elif k in ['rs1', 'rs2', 'rs3']:
      t = tkn(v, False)
      ret.append(t)
    elif k in ['imm', 'uimm', 'offset', 'shamt']:
      t =tkn(v)
      ret.append(t)
    else:
      assert False
  return ret
