#include "common.h"
#include "sopt.h"

#define GENFUNCS(T) \
    public constfunc T add(T a, T b) {return a + b;} \
    public constfunc T sub(T a, T b) {return a - b;} \
    public constfunc T mul(T a, T b) {return a * b;} \
    public constfunc T div(T a, T b) {return a / b;} \
    public constfunc T mod(T a, T b) {return a % b;} \
    public constfunc bool eq(T a, T b) {return a == b;} \
    public constfunc bool neq(T a, T b) {return a != b;} \
    public constfunc bool gt(T a, T b) {return a > b;} \
    public constfunc bool ge(T a, T b) {return a >= b;} \
    public constfunc bool lt(T a, T b) {return a < b;} \
    public constfunc bool le(T a, T b) {return a <= b;} \
    public constfunc T and(T a, T b){return a & b;} \
    public constfunc T or(T a, T b){return a | b;} \
    public constfunc T xor(T a, T b){return a ^ b;} \
    public constfunc T lsl(T a, T b){return a << b;} \
    public constfunc T lsr(T a, T b){return a >> b;} \
    public constfunc T asr(T a, T b){return a >> b;} \
    public constfunc T not(T a){return ~a;} \

typedef struct sopt
{
   s32 max_tokens;
} sopt;

void gen(int max_tokens, int num_vars)
{
    // gen enumerates all programs of #tokens 1 ... max_tokens
    // a program is an ast
}

int asdf(int a, int b, int c) {
    return (((((!1)==b)?2:((!a)>>(~a)))>(((~1)+(2+1))|((~3)+c)))?(2/b):((((3|1)+(~2))<=((c|3)<<(2^3)))?((!a)*3):(((((~b)<<b)%c)>2)?((!b)>>2):((!2)|(a>>(b<<(1&3)))))));
}

int main(int argc, char** argv)
{
    printf("%d\n",asdf(5,6,7));
}