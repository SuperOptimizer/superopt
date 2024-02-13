
#x86 currently uses a character level encoding scheme
#so 256 tokens for byte values, plus 2 for PAD and DECSTART

NUM_TOKENS = 258
METATOKENS = ['PAD','DECSTART']

def tkn(t: str):
    if t.startswith('0x'):
        return int(t, 16)
    elif t in METATOKENS:
        return 256 + METATOKENS.index(t)
    else:
        return int(t, 10)

def tokenize(prog: str, encoder: bool, ctxlen: int):
    '''prog is the output from objdump'''
    ret = []
    for line in prog.split('\n'):
        pass
