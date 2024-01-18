import random

arith_ops = ['+','-','*','/','%',]
cmp_ops = ['==','!=','>','<','>=','<=']
logical_ops = ['&&','||']
bit_ops = ['&','|','^','>>','<<']
ternary_ops = ['?']

ALL_BINOPS = arith_ops + cmp_ops + logical_ops + bit_ops

def gen_random(max_tokens, vars, constants):
    assert max_tokens > 0
    if max_tokens == 1:
        return random.choice(vars + constants)
    elif max_tokens == 2:
        op = random.choice(['~','!'])
        arg = random.choice(vars + constants)
        return '(' + op + arg + ')'
    elif max_tokens >= 3:
        op = random.choice(ALL_BINOPS)
        lhs_size = random.randint(1,max_tokens-2)
        rhs_size = max_tokens - 1 - lhs_size
        lhs = gen_random(lhs_size,vars,constants)
        rhs = gen_random(rhs_size,vars,constants)
        return f'({lhs}{op}{rhs})'



def gen_exhaustive(max_tokens, vars, constants):
    if max_tokens == 1:
        yield from vars
        yield from constants
    if max_tokens == 2:
        for op in ['~','!']:
            for t in gen_exhaustive(1,vars,constants):
                yield '(' + op + t + ')'
    if max_tokens >= 3:
        for op in ALL_BINOPS:
            for x in range(1,max_tokens-1):
                for lhs in gen_exhaustive(x,vars,constants):
                    for rhs in gen_exhaustive(max_tokens-1-x,vars,constants):
                        yield '(' + lhs + op + rhs + ')'

print(gen_random(7,['a','b','c'], ['1','2','3']))

