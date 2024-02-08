import random
import copy

arith_ops = ['+','-','*',] #'/','%',]
cmp_ops = ['==','!=','>','<','>=','<=']
logical_ops = ['&&','||']
bit_ops = ['&','|','^',] # '>>','<<']
ternary_ops = ['?']

MATH_BINOPS = arith_ops + bit_ops

toplevel_stmts = ['funcdef', '='] #'structdef','uniondef','enumdef',
embedded_stmts = ['for', 'while', '=', 'return'] #'goto'

#caller MUST add the var to global_vars
def new_global(global_vars):
  var = 'g_' + str(len(global_vars))
  return var

#caller MUST add the var to local_vars
def new_local(locals):
  var = 'l_' + str(len(locals))
  return var

def new_func(funcs):
  func = 'func_'+ str(len(funcs))
  return func

def gen_random_stmts(num_stmts, constants, tabs, global_vars, local_vars):
  depth = len(tabs)
  if depth >= 5:
    return []
  if num_stmts == 0:
    return []
  stmts = []

  while num_stmts > 0:
    if depth == 0:
      stmt_type = random.choice(toplevel_stmts)
      if stmt_type == '=':
        lhs = new_global(global_vars)
        rhs = gen_random_expr(16, global_vars+local_vars,constants)
        stmt = f'__auto_type {lhs} = {rhs};'
        global_vars.append(lhs)
        stmts.append(stmt)
        num_stmts -= 1
      elif stmt_type == 'funcdef':
        rettype = 'int' #todo: make better
        params = []
        for x in range(random.randint(1,4)):
          params.append(new_local(params))
        body = gen_random_stmts(num_stmts-1, constants,'\t', global_vars, copy.deepcopy(params))
    else:
      #we are in a function definition
      stmt_type = random.choice(embedded_stmts)
      if stmt_type == '=':
        lhs = new_local(local_vars)
        rhs = gen_random_expr(16, global_vars + local_vars, constants)
        local_vars.append(lhs)
        stmt = f'{tabs}__auto_type {lhs} = {rhs};'
      elif stmt_type == 'while':
        cmp_op = random.choice(cmp_ops)
        cmp_lhs = gen_random_expr(8, global_vars + local_vars, constants)
        cmp_rhs = gen_random_expr(8, global_vars + local_vars, constants)
        body = gen_random_stmts(num_stmts-1,constants, tabs + '\t',global_vars, local_vars)
        body_str = '\n'.join(body)
        stmt = f'{tabs}while({cmp_lhs}{cmp_op}{cmp_rhs}){body_str}'
      elif stmt_type == 'for':
        idx = new_local(local_vars)
        bound = random.choice(constants + local_vars + global_vars)
        body = gen_random_stmts(num_stmts-1,constants,tabs + '\t',global_vars, local_vars)
        body_str = '\n'.join(body)
        stmt = f'{tabs}for(int {idx} = 0; {idx} < {bound}; {idx}++){{{body_str}}}'


def gen_random_expr(max_tokens, vars, constants):
  assert max_tokens > 0
  if max_tokens == 1:
    return '(' + random.choice(vars + constants) + ')'
  elif max_tokens == 2:
    op = random.choice(['~','!'])
    arg = random.choice(vars + constants)
    return '(' + op + '(' + arg + '))'
  elif max_tokens >= 10:
    cmp = random.choice(cmp_ops)
    cmp_tkns = max_tokens //3
    cmp_lhs_size = random.randint(1,cmp_tkns-1)
    cmp_rhs_size = cmp_tkns - cmp_lhs_size
    cmp_lhs = gen_random_expr(cmp_lhs_size,vars,constants)
    cmp_rhs = gen_random_expr(cmp_rhs_size,vars,constants)

    max_tokens = max_tokens//3*2
    lhs_size = random.randint(1, max_tokens-1)
    rhs_size = max_tokens - lhs_size
    lhs = gen_random_expr(lhs_size,vars,constants)
    rhs = gen_random_expr(rhs_size,vars,constants)
    return f'(({cmp_lhs}{cmp}{cmp_rhs})?{lhs}:{rhs})'

  elif max_tokens >= 3:
    op = random.choice(MATH_BINOPS)
    lhs_size = random.randint(1,max_tokens-2)
    rhs_size = max_tokens - 1 - lhs_size
    lhs = gen_random_expr(lhs_size,vars,constants)
    rhs = gen_random_expr(rhs_size,vars,constants)
    return f'({lhs}{op}{rhs})'

def gen_random_func(max_tokens, args, constants, dtypes, name):
  body = gen_random_expr(max_tokens, args, constants)
  args_str = ','.join(' '.join([random.choice(dtypes), a]) for a in args)
  return f'{random.choice(dtypes)} {name}({args_str}){{ return {body};}}'

def gen_exhaustive_expr(max_tokens, vars, constants):
  if max_tokens == 1:
    yield from vars
    yield from constants
  if max_tokens == 2:
    for op in ['~','!']:
      for t in gen_exhaustive_expr(1,vars,constants):
        yield '(' + op + t + ')'
  if max_tokens >= 3:
    for op in MATH_BINOPS:
      for x in range(1,max_tokens-1):
        for lhs in gen_exhaustive_expr(x,vars,constants):
          for rhs in gen_exhaustive_expr(max_tokens-1-x,vars,constants):
            yield '(' + lhs + op + rhs + ')'

