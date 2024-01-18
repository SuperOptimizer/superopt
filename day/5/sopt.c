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


int main(int argc, char** argv)
{

}