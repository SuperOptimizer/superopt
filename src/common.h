#pragma once

#include <stdint.h>
#include <stdio.h>
//#include <stdlib.h> div_t grrr >:(
#include <stdbool.h>
#include <assert.h>
#include <memory.h>

//stdlib.h stuff
void *malloc(size_t size);
void free(void* ptr);
void *realloc(void *ptr, size_t size);


#define die(msg) do{printf("%s\n",msg); exit(1)}while(0)
#define die_if(cond,msg) do{if(cond){printf("%s\n",msg);exit(1);}}while(0)
#define warn(msg) do{printf("%s\n",msg);}while(0);
#define warn_if(cond,msg) do{if(cond){printf("%s\n",msg);}}while(0)

#define auto __auto_type
#define overload __attribute__((overloadable))
#define purefunc __attribute__((pure))
#define constfunc __attribute__((const))
#define hotfunc __attribute__((hot))
#define coldfunc __attribute__((cold))
#define leaffunc __attribute__((leaf))
#define aligned(n) __attribute__((aligned(n)))
#define cleanup(func) __attribute__((cleanup(func)))
#define fallthrough __attribute__((fallthrough))
#define assume(stmt) __attribute__((assume(stmt)))
#define unreachable() __builtin_unreachable()

#define public static inline overload __attribute__((visibility("default")))
#define private static inline overload __attribute__((visibility("hidden")))

typedef uint8_t u8;
typedef int8_t s8;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;
typedef  __uint128_t u128;
typedef  __int128_t s128;
typedef _Float16 f16;
typedef float f32;
typedef double f64;

typedef union r8 {
	u8 u8[1];
	s8 s8[1];
} r8;

typedef union r16 {
	u8 u8[2];
	s8 s8[2];
	u16 u16[1];
	s16 s16[1];
	f16 f16[1];
} r16;

typedef union r32 {
	u8 u8[4];
	s8 s8[4];
	u16 u16[2];
	s16 s16[2];
	u32 u32[1];
	s32 s32[1];
	f16 f16[2];
	f32 f32[1];
} r32;


typedef union r64 {
	u8 u8[8];
	s8 s8[8];
	u16 u16[4];
	s16 s16[4];
	u32 u32[2];
	s32 s32[2];
	u64 u64[1];
	s64 s64[1];
	f16 f16[4];
	f32 f32[2];
	f64 f64[1];
} r64;


typedef union r128 {
	u8 u8[16];
	s8 s8[16];
	u16 u16[8];
	s16 s16[8];
	u32 u32[4];
	s32 s32[4];
	u64 u64[2];
	s64 s64[2];
	u128 u128[1];
	s128 s128[1];
	f16 f16[8];
	f32 f32[4];
	f64 f64[2];
} r128;

typedef enum dtype {
	DT_S8 = 0,
	DT_U8,
	DT_S16,
	DT_U16,
	DT_S32,
	DT_U32,
	DT_S64,
	DT_U64,
	DT_S128,
	DT_U128,
	DT_F16,
	DT_F32,
	DT_F64
} dtype;


// https://prng.di.unimi.it/
/* todo: does a public static function with static function data have multiple copies of said static data in each */
/* implementation in different translation units? */
public u64 randint(void) {
#define rotl(x, k) ((x << k) | (x >> (64 - k)))
	static u8 s[4] = {0x1234567890abcdefULL, 0xfedcba0987654321ULL, 0x1f2e3d4c5b6a7089ULL, 0xdeadbeefcafebabeULL};
	const u8 result = rotl(s[1] * 5, 7) * 9;
	const u8 t = s[1] << 17;
	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = rotl(s[3], 45);
	return result;
#undef rotl
}


#define vec(n,t) t __attribute__((ext_vector_type(n)))
#define mat(n,m,t) t __attribute__((matrix_type(n,m)))

typedef vec(1,u8) u8x1;
typedef vec(2,u8) u8x2;
typedef vec(3,u8) u8x3;
typedef vec(4,u8) u8x4;
typedef vec(8,u8) u8x8;
typedef vec(16,u8) u8x16;
typedef vec(32,u8) u8x32;
typedef vec(64,u8) u8x64;
typedef vec(128,u8) u8x128;
typedef vec(1,s8) s8x1;
typedef vec(2,s8) s8x2;
typedef vec(3,s8) s8x3;
typedef vec(4,s8) s8x4;
typedef vec(8,s8) s8x8;
typedef vec(16,s8) s8x16;
typedef vec(32,s8) s8x32;
typedef vec(64,s8) s8x64;
typedef vec(128,s8) s8x128;

typedef vec(1,u16) u16x1;
typedef vec(2,u16) u16x2;
typedef vec(3,u16) u16x3;
typedef vec(4,u16) u16x4;
typedef vec(8,u16) u16x8;
typedef vec(16,u16) u16x16;
typedef vec(32,u16) u16x32;
typedef vec(64,u16) u16x64;
typedef vec(1,s16) s16x1;
typedef vec(2,s16) s16x2;
typedef vec(3,s16) s16x3;
typedef vec(4,s16) s16x4;
typedef vec(8,s16) s16x8;
typedef vec(16,s16) s16x16;
typedef vec(32,s16) s16x32;
typedef vec(64,s16) s16x64;

typedef vec(1,u32) u32x1;
typedef vec(2,u32) u32x2;
typedef vec(3,u32) u32x3;
typedef vec(4,u32) u32x4;
typedef vec(8,u32) u32x8;
typedef vec(16,u32) u32x16;
typedef vec(32,u32) u32x32;
typedef vec(1,s32) s32x1;
typedef vec(2,s32) s32x2;
typedef vec(3,s32) s32x3;
typedef vec(4,s32) s32x4;
typedef vec(8,s32) s32x8;
typedef vec(16,s32) s32x16;
typedef vec(32,s32) s32x32;

typedef vec(1,u64) u64x1;
typedef vec(2,u64) u64x2;
typedef vec(3,u64) u64x3;
typedef vec(4,u64) u64x4;
typedef vec(8,u64) u64x8;
typedef vec(16,u64) u64x16;
typedef vec(1,s64) s64x1;
typedef vec(2,s64) s64x2;
typedef vec(3,s64) s64x3;
typedef vec(4,s64) s64x4;
typedef vec(8,s64) s64x8;
typedef vec(16,s64) s64x16;

typedef vec(1,u128) u128x1;
typedef vec(2,u128) u128x2;
typedef vec(3,u128) u128x3;
typedef vec(4,u128) u128x4;
typedef vec(8,u128) u128x8;
typedef vec(1,s128) s128x1;
typedef vec(2,s128) s128x2;
typedef vec(3,s128) s128x3;
typedef vec(4,s128) s128x4;
typedef vec(8,s128) s128x8;
typedef vec(16,s128) s128x16;

typedef vec(1,f16) f16x1;
typedef vec(2,f16) f16x2;
typedef vec(3,f16) f16x3;
typedef vec(4,f16) f16x4;
typedef vec(8,f16) f16x8;
typedef vec(16,f16) f16x16;
typedef vec(32,f16) f16x32;
typedef vec(64,f16) f16x64;
typedef vec(1,f32) f32x1;
typedef vec(2,f32) f32x2;
typedef vec(3,f32) f32x3;
typedef vec(4,f32) f32x4;
typedef vec(8,f32) f32x8;
typedef vec(16,f32) f32x16;
typedef vec(32,f64) f32x32;
typedef vec(1,f64) f64x1;
typedef vec(2,f64) f64x2;
typedef vec(3,f64) f64x3;
typedef vec(4,f64) f64x4;
typedef vec(8,f64) f64x8;
typedef vec(16,f64) f64x16;



typedef mat(1,1,u8) u8x1x1;
typedef mat(1,2,u8) u8x1x2;
typedef mat(1,4,u8) u8x1x4;
typedef mat(1,8,u8) u8x1x8;
typedef mat(1,16,u8) u8x1x16;
typedef mat(1,32,u8) u8x1x32;
typedef mat(1,64,u8) u8x1x64;
typedef mat(1,128,u8) u8x1x128;

typedef mat(2,1,u8) u8x2x1;
typedef mat(2,2,u8) u8x2x2;
typedef mat(2,4,u8) u8x2x4;
typedef mat(2,8,u8) u8x2x8;
typedef mat(2,16,u8) u8x2x16;
typedef mat(2,32,u8) u8x2x32;
typedef mat(2,64,u8) u8x2x64;
typedef mat(2,128,u8) u8x2x128;

typedef mat(4,1,u8) u8x4x1;
typedef mat(4,2,u8) u8x4x2;
typedef mat(4,4,u8) u8x4x4;
typedef mat(4,8,u8) u8x4x8;
typedef mat(4,16,u8) u8x4x16;
typedef mat(4,32,u8) u8x4x32;
typedef mat(4,64,u8) u8x4x64;
typedef mat(4,128,u8) u8x4x128;

typedef mat(8,1,u8) u8x8x1;
typedef mat(8,2,u8) u8x8x2;
typedef mat(8,4,u8) u8x8x4;
typedef mat(8,8,u8) u8x8x8;
typedef mat(8,16,u8) u8x8x16;
typedef mat(8,32,u8) u8x8x32;
typedef mat(8,64,u8) u8x8x64;
typedef mat(8,128,u8) u8x8x128;


typedef mat(16,1,u8) u8x16x1;
typedef mat(16,2,u8) u8x16x2;
typedef mat(16,4,u8) u8x16x4;
typedef mat(16,8,u8) u8x16x8;
typedef mat(16,16,u8) u8x16x16;
typedef mat(16,32,u8) u8x16x32;
typedef mat(16,64,u8) u8x16x64;
typedef mat(16,128,u8) u8x16x128;

typedef mat(32,1,u8) u8x32x1;
typedef mat(32,2,u8) u8x32x2;
typedef mat(32,4,u8) u8x32x4;
typedef mat(32,8,u8) u8x32x8;
typedef mat(32,16,u8) u8x32x16;
typedef mat(32,32,u8) u8x32x32;
typedef mat(32,64,u8) u8x32x64;
typedef mat(32,128,u8) u8x32x128;

typedef mat(64,1,u8) u8x64x1;
typedef mat(64,2,u8) u8x64x2;
typedef mat(64,4,u8) u8x64x4;
typedef mat(64,8,u8) u8x64x8;
typedef mat(64,16,u8) u8x64x16;
typedef mat(64,32,u8) u8x64x32;
typedef mat(64,64,u8) u8x64x64;
typedef mat(64,128,u8) u8x64x128;

typedef mat(128,1,u8) u8x128x1;
typedef mat(128,2,u8) u8x128x2;
typedef mat(128,4,u8) u8x128x4;
typedef mat(128,8,u8) u8x128x8;
typedef mat(128,16,u8) u8x128x16;
typedef mat(128,32,u8) u8x128x32;
typedef mat(128,64,u8) u8x128x64;
typedef mat(128,128,u8) u8x128x128;


typedef mat(1,1,s8) s8x1x1;
typedef mat(1,2,s8) s8x1x2;
typedef mat(1,4,s8) s8x1x4;
typedef mat(1,8,s8) s8x1x8;
typedef mat(1,16,s8) s8x1x16;
typedef mat(1,32,s8) s8x1x32;
typedef mat(1,64,s8) s8x1x64;
typedef mat(1,128,s8) s8x1x128;

typedef mat(2,1,s8) s8x2x1;
typedef mat(2,2,s8) s8x2x2;
typedef mat(2,4,s8) s8x2x4;
typedef mat(2,8,s8) s8x2x8;
typedef mat(2,16,s8) s8x2x16;
typedef mat(2,32,s8) s8x2x32;
typedef mat(2,64,s8) s8x2x64;
typedef mat(2,128,s8) s8x2x128;

typedef mat(4,1,s8) s8x4x1;
typedef mat(4,2,s8) s8x4x2;
typedef mat(4,4,s8) s8x4x4;
typedef mat(4,8,s8) s8x4x8;
typedef mat(4,16,s8) s8x4x16;
typedef mat(4,32,s8) s8x4x32;
typedef mat(4,64,s8) s8x4x64;
typedef mat(4,128,s8) s8x4x128;

typedef mat(8,1,s8) s8x8x1;
typedef mat(8,2,s8) s8x8x2;
typedef mat(8,4,s8) s8x8x4;
typedef mat(8,8,s8) s8x8x8;
typedef mat(8,16,s8) s8x8x16;
typedef mat(8,32,s8) s8x8x32;
typedef mat(8,64,s8) s8x8x64;
typedef mat(8,128,s8) s8x8x128;

typedef mat(16,1,s8) s8x16x1;
typedef mat(16,2,s8) s8x16x2;
typedef mat(16,4,s8) s8x16x4;
typedef mat(16,8,s8) s8x16x8;
typedef mat(16,16,s8) s8x16x16;
typedef mat(16,32,s8) s8x16x32;
typedef mat(16,64,s8) s8x16x64;
typedef mat(16,128,s8) s8x16x128;

typedef mat(32,1,s8) s8x32x1;
typedef mat(32,2,s8) s8x32x2;
typedef mat(32,4,s8) s8x32x4;
typedef mat(32,8,s8) s8x32x8;
typedef mat(32,16,s8) s8x32x16;
typedef mat(32,32,s8) s8x32x32;
typedef mat(32,64,s8) s8x32x64;
typedef mat(32,128,s8) s8x32x128;

typedef mat(64,1,s8) s8x64x1;
typedef mat(64,2,s8) s8x64x2;
typedef mat(64,4,s8) s8x64x4;
typedef mat(64,8,s8) s8x64x8;
typedef mat(64,16,s8) s8x64x16;
typedef mat(64,32,s8) s8x64x32;
typedef mat(64,64,s8) s8x64x64;
typedef mat(64,128,s8) s8x64x128;

typedef mat(128,1,s8) s8x128x1;
typedef mat(128,2,s8) s8x128x2;
typedef mat(128,4,s8) s8x128x4;
typedef mat(128,8,s8) s8x128x8;
typedef mat(128,16,s8) s8x128x16;
typedef mat(128,32,s8) s8x128x32;
typedef mat(128,64,s8) s8x128x64;
typedef mat(128,128,s8) s8x128x128;


typedef mat(1,1,u16) u16x1x1;
typedef mat(1,2,u16) u16x1x2;
typedef mat(1,4,u16) u16x1x4;
typedef mat(1,8,u16) u16x1x8;
typedef mat(1,16,u16) u16x1x16;
typedef mat(1,32,u16) u16x1x32;
typedef mat(1,64,u16) u16x1x64;
typedef mat(1,128,u16) u16x1x128;

typedef mat(2,1,u16) u16x2x1;
typedef mat(2,2,u16) u16x2x2;
typedef mat(2,4,u16) u16x2x4;
typedef mat(2,8,u16) u16x2x8;
typedef mat(2,16,u16) u16x2x16;
typedef mat(2,32,u16) u16x2x32;
typedef mat(2,64,u16) u16x2x64;
typedef mat(2,128,u16) u16x2x128;

typedef mat(4,1,u16) u16x4x1;
typedef mat(4,2,u16) u16x4x2;
typedef mat(4,4,u16) u16x4x4;
typedef mat(4,8,u16) u16x4x8;
typedef mat(4,16,u16) u16x4x16;
typedef mat(4,32,u16) u16x4x32;
typedef mat(4,64,u16) u16x4x64;
typedef mat(4,128,u16) u16x4x128;

typedef mat(8,1,u16) u16x8x1;
typedef mat(8,2,u16) u16x8x2;
typedef mat(8,4,u16) u16x8x4;
typedef mat(8,8,u16) u16x8x8;
typedef mat(8,16,u16) u16x8x16;
typedef mat(8,32,u16) u16x8x32;
typedef mat(8,64,u16) u16x8x64;
typedef mat(8,128,u16) u16x8x128;

typedef mat(16,1,u16) u16x16x1;
typedef mat(16,2,u16) u16x16x2;
typedef mat(16,4,u16) u16x16x4;
typedef mat(16,8,u16) u16x16x8;
typedef mat(16,16,u16) u16x16x16;
typedef mat(16,32,u16) u16x16x32;
typedef mat(16,64,u16) u16x16x64;
typedef mat(16,128,u16) u16x16x128;

typedef mat(32,1,u16) u16x32x1;
typedef mat(32,2,u16) u16x32x2;
typedef mat(32,4,u16) u16x32x4;
typedef mat(32,8,u16) u16x32x8;
typedef mat(32,16,u16) u16x32x16;
typedef mat(32,32,u16) u16x32x32;
typedef mat(32,64,u16) u16x32x64;
typedef mat(32,128,u16) u16x32x128;

typedef mat(64,1,u16) u16x64x1;
typedef mat(64,2,u16) u16x64x2;
typedef mat(64,4,u16) u16x64x4;
typedef mat(64,8,u16) u16x64x8;
typedef mat(64,16,u16) u16x64x16;
typedef mat(64,32,u16) u16x64x32;
typedef mat(64,64,u16) u16x64x64;
typedef mat(64,128,u16) u16x64x128;

typedef mat(128,1,u16) u16x128x1;
typedef mat(128,2,u16) u16x128x2;
typedef mat(128,4,u16) u16x128x4;
typedef mat(128,8,u16) u16x128x8;
typedef mat(128,16,u16) u16x128x16;
typedef mat(128,32,u16) u16x128x32;
typedef mat(128,64,u16) u16x128x64;
typedef mat(128,128,u16) u16x128x128;


typedef mat(1,1,s16) s16x1x1;
typedef mat(1,2,s16) s16x1x2;
typedef mat(1,4,s16) s16x1x4;
typedef mat(1,8,s16) s16x1x8;
typedef mat(1,16,s16) s16x1x16;
typedef mat(1,32,s16) s16x1x32;
typedef mat(1,64,s16) s16x1x64;
typedef mat(1,128,s16) s16x1x128;

typedef mat(2,1,s16) s16x2x1;
typedef mat(2,2,s16) s16x2x2;
typedef mat(2,4,s16) s16x2x4;
typedef mat(2,8,s16) s16x2x8;
typedef mat(2,16,s16) s16x2x16;
typedef mat(2,32,s16) s16x2x32;
typedef mat(2,64,s16) s16x2x64;
typedef mat(2,128,s16) s16x2x128;

typedef mat(4,1,s16) s16x4x1;
typedef mat(4,2,s16) s16x4x2;
typedef mat(4,4,s16) s16x4x4;
typedef mat(4,8,s16) s16x4x8;
typedef mat(4,16,s16) s16x4x16;
typedef mat(4,32,s16) s16x4x32;
typedef mat(4,64,s16) s16x4x64;
typedef mat(4,128,s16) s16x4x128;

typedef mat(8,1,s16) s16x8x1;
typedef mat(8,2,s16) s16x8x2;
typedef mat(8,4,s16) s16x8x4;
typedef mat(8,8,s16) s16x8x8;
typedef mat(8,16,s16) s16x8x16;
typedef mat(8,32,s16) s16x8x32;
typedef mat(8,64,s16) s16x8x64;
typedef mat(8,128,s16) s16x8x128;

typedef mat(16,1,s16) s16x16x1;
typedef mat(16,2,s16) s16x16x2;
typedef mat(16,4,s16) s16x16x4;
typedef mat(16,8,s16) s16x16x8;
typedef mat(16,16,s16) s16x16x16;
typedef mat(16,32,s16) s16x16x32;
typedef mat(16,64,s16) s16x16x64;
typedef mat(16,128,s16) s16x16x128;

typedef mat(32,1,s16) s16x32x1;
typedef mat(32,2,s16) s16x32x2;
typedef mat(32,4,s16) s16x32x4;
typedef mat(32,8,s16) s16x32x8;
typedef mat(32,16,s16) s16x32x16;
typedef mat(32,32,s16) s16x32x32;
typedef mat(32,64,s16) s16x32x64;
typedef mat(32,128,s16) s16x32x128;

typedef mat(64,1,s16) s16x64x1;
typedef mat(64,2,s16) s16x64x2;
typedef mat(64,4,s16) s16x64x4;
typedef mat(64,8,s16) s16x64x8;
typedef mat(64,16,s16) s16x64x16;
typedef mat(64,32,s16) s16x64x32;
typedef mat(64,64,s16) s16x64x64;
typedef mat(64,128,s16) s16x64x128;

typedef mat(128,1,s16) s16x128x1;
typedef mat(128,2,s16) s16x128x2;
typedef mat(128,4,s16) s16x128x4;
typedef mat(128,8,s16) s16x128x8;
typedef mat(128,16,s16) s16x128x16;
typedef mat(128,32,s16) s16x128x32;
typedef mat(128,64,s16) s16x128x64;
typedef mat(128,128,s16) s16x128x128;


typedef mat(1,1,u32) u32x1x1;
typedef mat(1,2,u32) u32x1x2;
typedef mat(1,4,u32) u32x1x4;
typedef mat(1,8,u32) u32x1x8;
typedef mat(1,16,u32) u32x1x16;
typedef mat(1,32,u32) u32x1x32;
typedef mat(1,64,u32) u32x1x64;
typedef mat(1,128,u32) u32x1x128;

typedef mat(2,1,u32) u32x2x1;
typedef mat(2,2,u32) u32x2x2;
typedef mat(2,4,u32) u32x2x4;
typedef mat(2,8,u32) u32x2x8;
typedef mat(2,16,u32) u32x2x16;
typedef mat(2,32,u32) u32x2x32;
typedef mat(2,64,u32) u32x2x64;
typedef mat(2,128,u32) u32x2x128;

typedef mat(4,1,u32) u32x4x1;
typedef mat(4,2,u32) u32x4x2;
typedef mat(4,4,u32) u32x4x4;
typedef mat(4,8,u32) u32x4x8;
typedef mat(4,16,u32) u32x4x16;
typedef mat(4,32,u32) u32x4x32;
typedef mat(4,64,u32) u32x4x64;
typedef mat(4,128,u32) u32x4x128;

typedef mat(8,1,u32) u32x8x1;
typedef mat(8,2,u32) u32x8x2;
typedef mat(8,4,u32) u32x8x4;
typedef mat(8,8,u32) u32x8x8;
typedef mat(8,16,u32) u32x8x16;
typedef mat(8,32,u32) u32x8x32;
typedef mat(8,64,u32) u32x8x64;
typedef mat(8,128,u32) u32x8x128;

typedef mat(16,1,u32) u32x16x1;
typedef mat(16,2,u32) u32x16x2;
typedef mat(16,4,u32) u32x16x4;
typedef mat(16,8,u32) u32x16x8;
typedef mat(16,16,u32) u32x16x16;
typedef mat(16,32,u32) u32x16x32;
typedef mat(16,64,u32) u32x16x64;
typedef mat(16,128,u32) u32x16x128;

typedef mat(32,1,u32) u32x32x1;
typedef mat(32,2,u32) u32x32x2;
typedef mat(32,4,u32) u32x32x4;
typedef mat(32,8,u32) u32x32x8;
typedef mat(32,16,u32) u32x32x16;
typedef mat(32,32,u32) u32x32x32;
typedef mat(32,64,u32) u32x32x64;
typedef mat(32,128,u32) u32x32x128;

typedef mat(64,1,u32) u32x64x1;
typedef mat(64,2,u32) u32x64x2;
typedef mat(64,4,u32) u32x64x4;
typedef mat(64,8,u32) u32x64x8;
typedef mat(64,16,u32) u32x64x16;
typedef mat(64,32,u32) u32x64x32;
typedef mat(64,64,u32) u32x64x64;
typedef mat(64,128,u32) u32x64x128;

typedef mat(128,1,u32) u32x128x1;
typedef mat(128,2,u32) u32x128x2;
typedef mat(128,4,u32) u32x128x4;
typedef mat(128,8,u32) u32x128x8;
typedef mat(128,16,u32) u32x128x16;
typedef mat(128,32,u32) u32x128x32;
typedef mat(128,64,u32) u32x128x64;
typedef mat(128,128,u32) u32x128x128;


typedef mat(1,1,s32) s32x1x1;
typedef mat(1,2,s32) s32x1x2;
typedef mat(1,4,s32) s32x1x4;
typedef mat(1,8,s32) s32x1x8;
typedef mat(1,16,s32) s32x1x16;
typedef mat(1,32,s32) s32x1x32;
typedef mat(1,64,s32) s32x1x64;
typedef mat(1,128,s32) s32x1x128;

typedef mat(2,1,s32) s32x2x1;
typedef mat(2,2,s32) s32x2x2;
typedef mat(2,4,s32) s32x2x4;
typedef mat(2,8,s32) s32x2x8;
typedef mat(2,16,s32) s32x2x16;
typedef mat(2,32,s32) s32x2x32;
typedef mat(2,64,s32) s32x2x64;
typedef mat(2,128,s32) s32x2x128;

typedef mat(4,1,s32) s32x4x1;
typedef mat(4,2,s32) s32x4x2;
typedef mat(4,4,s32) s32x4x4;
typedef mat(4,8,s32) s32x4x8;
typedef mat(4,16,s32) s32x4x16;
typedef mat(4,32,s32) s32x4x32;
typedef mat(4,64,s32) s32x4x64;
typedef mat(4,128,s32) s32x4x128;

typedef mat(8,1,s32) s32x8x1;
typedef mat(8,2,s32) s32x8x2;
typedef mat(8,4,s32) s32x8x4;
typedef mat(8,8,s32) s32x8x8;
typedef mat(8,16,s32) s32x8x16;
typedef mat(8,32,s32) s32x8x32;
typedef mat(8,64,s32) s32x8x64;
typedef mat(8,128,s32) s32x8x128;

typedef mat(16,1,s32) s32x16x1;
typedef mat(16,2,s32) s32x16x2;
typedef mat(16,4,s32) s32x16x4;
typedef mat(16,8,s32) s32x16x8;
typedef mat(16,16,s32) s32x16x16;
typedef mat(16,32,s32) s32x16x32;
typedef mat(16,64,s32) s32x16x64;
typedef mat(16,128,s32) s32x16x128;

typedef mat(32,1,s32) s32x32x1;
typedef mat(32,2,s32) s32x32x2;
typedef mat(32,4,s32) s32x32x4;
typedef mat(32,8,s32) s32x32x8;
typedef mat(32,16,s32) s32x32x16;
typedef mat(32,32,s32) s32x32x32;
typedef mat(32,64,s32) s32x32x64;
typedef mat(32,128,s32) s32x32x128;

typedef mat(64,1,s32) s32x64x1;
typedef mat(64,2,s32) s32x64x2;
typedef mat(64,4,s32) s32x64x4;
typedef mat(64,8,s32) s32x64x8;
typedef mat(64,16,s32) s32x64x16;
typedef mat(64,32,s32) s32x64x32;
typedef mat(64,64,s32) s32x64x64;
typedef mat(64,128,s32) s32x64x128;

typedef mat(128,1,s32) s32x128x1;
typedef mat(128,2,s32) s32x128x2;
typedef mat(128,4,s32) s32x128x4;
typedef mat(128,8,s32) s32x128x8;
typedef mat(128,16,s32) s32x128x16;
typedef mat(128,32,s32) s32x128x32;
typedef mat(128,64,s32) s32x128x64;
typedef mat(128,128,s32) s32x128x128;


typedef mat(1,1,u64) u64x1x1;
typedef mat(1,2,u64) u64x1x2;
typedef mat(1,4,u64) u64x1x4;
typedef mat(1,8,u64) u64x1x8;
typedef mat(1,16,u64) u64x1x16;
typedef mat(1,32,u64) u64x1x32;
typedef mat(1,64,u64) u64x1x64;
typedef mat(1,128,u64) u64x1x128;

typedef mat(2,1,u64) u64x2x1;
typedef mat(2,2,u64) u64x2x2;
typedef mat(2,4,u64) u64x2x4;
typedef mat(2,8,u64) u64x2x8;
typedef mat(2,16,u64) u64x2x16;
typedef mat(2,32,u64) u64x2x32;
typedef mat(2,64,u64) u64x2x64;
typedef mat(2,128,u64) u64x2x128;

typedef mat(4,1,u64) u64x4x1;
typedef mat(4,2,u64) u64x4x2;
typedef mat(4,4,u64) u64x4x4;
typedef mat(4,8,u64) u64x4x8;
typedef mat(4,16,u64) u64x4x16;
typedef mat(4,32,u64) u64x4x32;
typedef mat(4,64,u64) u64x4x64;
typedef mat(4,128,u64) u64x4x128;

typedef mat(8,1,u64) u64x8x1;
typedef mat(8,2,u64) u64x8x2;
typedef mat(8,4,u64) u64x8x4;
typedef mat(8,8,u64) u64x8x8;
typedef mat(8,16,u64) u64x8x16;
typedef mat(8,32,u64) u64x8x32;
typedef mat(8,64,u64) u64x8x64;
typedef mat(8,128,u64) u64x8x128;

typedef mat(16,1,u64) u64x16x1;
typedef mat(16,2,u64) u64x16x2;
typedef mat(16,4,u64) u64x16x4;
typedef mat(16,8,u64) u64x16x8;
typedef mat(16,16,u64) u64x16x16;
typedef mat(16,32,u64) u64x16x32;
typedef mat(16,64,u64) u64x16x64;
typedef mat(16,128,u64) u64x16x128;

typedef mat(32,1,u64) u64x32x1;
typedef mat(32,2,u64) u64x32x2;
typedef mat(32,4,u64) u64x32x4;
typedef mat(32,8,u64) u64x32x8;
typedef mat(32,16,u64) u64x32x16;
typedef mat(32,32,u64) u64x32x32;
typedef mat(32,64,u64) u64x32x64;
typedef mat(32,128,u64) u64x32x128;

typedef mat(64,1,u64) u64x64x1;
typedef mat(64,2,u64) u64x64x2;
typedef mat(64,4,u64) u64x64x4;
typedef mat(64,8,u64) u64x64x8;
typedef mat(64,16,u64) u64x64x16;
typedef mat(64,32,u64) u64x64x32;
typedef mat(64,64,u64) u64x64x64;
typedef mat(64,128,u64) u64x64x128;

typedef mat(128,1,u64) u64x128x1;
typedef mat(128,2,u64) u64x128x2;
typedef mat(128,4,u64) u64x128x4;
typedef mat(128,8,u64) u64x128x8;
typedef mat(128,16,u64) u64x128x16;
typedef mat(128,32,u64) u64x128x32;
typedef mat(128,64,u64) u64x128x64;
typedef mat(128,128,u64) u64x128x128;


typedef mat(1,1,s64) s64x1x1;
typedef mat(1,2,s64) s64x1x2;
typedef mat(1,4,s64) s64x1x4;
typedef mat(1,8,s64) s64x1x8;
typedef mat(1,16,s64) s64x1x16;
typedef mat(1,32,s64) s64x1x32;
typedef mat(1,64,s64) s64x1x64;
typedef mat(1,128,s64) s64x1x128;

typedef mat(2,1,s64) s64x2x1;
typedef mat(2,2,s64) s64x2x2;
typedef mat(2,4,s64) s64x2x4;
typedef mat(2,8,s64) s64x2x8;
typedef mat(2,16,s64) s64x2x16;
typedef mat(2,32,s64) s64x2x32;
typedef mat(2,64,s64) s64x2x64;
typedef mat(2,128,s64) s64x2x128;

typedef mat(4,1,s64) s64x4x1;
typedef mat(4,2,s64) s64x4x2;
typedef mat(4,4,s64) s64x4x4;
typedef mat(4,8,s64) s64x4x8;
typedef mat(4,16,s64) s64x4x16;
typedef mat(4,32,s64) s64x4x32;
typedef mat(4,64,s64) s64x4x64;
typedef mat(4,128,s64) s64x4x128;

typedef mat(8,1,s64) s64x8x1;
typedef mat(8,2,s64) s64x8x2;
typedef mat(8,4,s64) s64x8x4;
typedef mat(8,8,s64) s64x8x8;
typedef mat(8,16,s64) s64x8x16;
typedef mat(8,32,s64) s64x8x32;
typedef mat(8,64,s64) s64x8x64;
typedef mat(8,128,s64) s64x8x128;

typedef mat(16,1,s64) s64x16x1;
typedef mat(16,2,s64) s64x16x2;
typedef mat(16,4,s64) s64x16x4;
typedef mat(16,8,s64) s64x16x8;
typedef mat(16,16,s64) s64x16x16;
typedef mat(16,32,s64) s64x16x32;
typedef mat(16,64,s64) s64x16x64;
typedef mat(16,128,s64) s64x16x128;

typedef mat(32,1,s64) s64x32x1;
typedef mat(32,2,s64) s64x32x2;
typedef mat(32,4,s64) s64x32x4;
typedef mat(32,8,s64) s64x32x8;
typedef mat(32,16,s64) s64x32x16;
typedef mat(32,32,s64) s64x32x32;
typedef mat(32,64,s64) s64x32x64;
typedef mat(32,128,s64) s64x32x128;

typedef mat(64,1,s64) s64x64x1;
typedef mat(64,2,s64) s64x64x2;
typedef mat(64,4,s64) s64x64x4;
typedef mat(64,8,s64) s64x64x8;
typedef mat(64,16,s64) s64x64x16;
typedef mat(64,32,s64) s64x64x32;
typedef mat(64,64,s64) s64x64x64;
typedef mat(64,128,s64) s64x64x128;

typedef mat(128,1,s64) s64x128x1;
typedef mat(128,2,s64) s64x128x2;
typedef mat(128,4,s64) s64x128x4;
typedef mat(128,8,s64) s64x128x8;
typedef mat(128,16,s64) s64x128x16;
typedef mat(128,32,s64) s64x128x32;
typedef mat(128,64,s64) s64x128x64;
typedef mat(128,128,s64) s64x128x128;


typedef mat(1,1,u128) u128x1x1;
typedef mat(1,2,u128) u128x1x2;
typedef mat(1,4,u128) u128x1x4;
typedef mat(1,8,u128) u128x1x8;
typedef mat(1,16,u128) u128x1x16;
typedef mat(1,32,u128) u128x1x32;
typedef mat(1,64,u128) u128x1x64;
typedef mat(1,128,u128) u128x1x128;

typedef mat(2,1,u128) u128x2x1;
typedef mat(2,2,u128) u128x2x2;
typedef mat(2,4,u128) u128x2x4;
typedef mat(2,8,u128) u128x2x8;
typedef mat(2,16,u128) u128x2x16;
typedef mat(2,32,u128) u128x2x32;
typedef mat(2,64,u128) u128x2x64;
typedef mat(2,128,u128) u128x2x128;

typedef mat(4,1,u128) u128x4x1;
typedef mat(4,2,u128) u128x4x2;
typedef mat(4,4,u128) u128x4x4;
typedef mat(4,8,u128) u128x4x8;
typedef mat(4,16,u128) u128x4x16;
typedef mat(4,32,u128) u128x4x32;
typedef mat(4,64,u128) u128x4x64;
typedef mat(4,128,u128) u128x4x128;

typedef mat(8,1,u128) u128x8x1;
typedef mat(8,2,u128) u128x8x2;
typedef mat(8,4,u128) u128x8x4;
typedef mat(8,8,u128) u128x8x8;
typedef mat(8,16,u128) u128x8x16;
typedef mat(8,32,u128) u128x8x32;
typedef mat(8,64,u128) u128x8x64;
typedef mat(8,128,u128) u128x8x128;

typedef mat(16,1,u128) u128x16x1;
typedef mat(16,2,u128) u128x16x2;
typedef mat(16,4,u128) u128x16x4;
typedef mat(16,8,u128) u128x16x8;
typedef mat(16,16,u128) u128x16x16;
typedef mat(16,32,u128) u128x16x32;
typedef mat(16,64,u128) u128x16x64;
typedef mat(16,128,u128) u128x16x128;

typedef mat(32,1,u128) u128x32x1;
typedef mat(32,2,u128) u128x32x2;
typedef mat(32,4,u128) u128x32x4;
typedef mat(32,8,u128) u128x32x8;
typedef mat(32,16,u128) u128x32x16;
typedef mat(32,32,u128) u128x32x32;
typedef mat(32,64,u128) u128x32x64;
typedef mat(32,128,u128) u128x32x128;

typedef mat(64,1,u128) u128x64x1;
typedef mat(64,2,u128) u128x64x2;
typedef mat(64,4,u128) u128x64x4;
typedef mat(64,8,u128) u128x64x8;
typedef mat(64,16,u128) u128x64x16;
typedef mat(64,32,u128) u128x64x32;
typedef mat(64,64,u128) u128x64x64;
typedef mat(64,128,u128) u128x64x128;

typedef mat(128,1,u128) u128x128x1;
typedef mat(128,2,u128) u128x128x2;
typedef mat(128,4,u128) u128x128x4;
typedef mat(128,8,u128) u128x128x8;
typedef mat(128,16,u128) u128x128x16;
typedef mat(128,32,u128) u128x128x32;
typedef mat(128,64,u128) u128x128x64;
typedef mat(128,128,u128) u128x128x128;


typedef mat(1,1,s128) s128x1x1;
typedef mat(1,2,s128) s128x1x2;
typedef mat(1,4,s128) s128x1x4;
typedef mat(1,8,s128) s128x1x8;
typedef mat(1,16,s128) s128x1x16;
typedef mat(1,32,s128) s128x1x32;
typedef mat(1,64,s128) s128x1x64;
typedef mat(1,128,s128) s128x1x128;

typedef mat(2,1,s128) s128x2x1;
typedef mat(2,2,s128) s128x2x2;
typedef mat(2,4,s128) s128x2x4;
typedef mat(2,8,s128) s128x2x8;
typedef mat(2,16,s128) s128x2x16;
typedef mat(2,32,s128) s128x2x32;
typedef mat(2,64,s128) s128x2x64;
typedef mat(2,128,s128) s128x2x128;

typedef mat(4,1,s128) s128x4x1;
typedef mat(4,2,s128) s128x4x2;
typedef mat(4,4,s128) s128x4x4;
typedef mat(4,8,s128) s128x4x8;
typedef mat(4,16,s128) s128x4x16;
typedef mat(4,32,s128) s128x4x32;
typedef mat(4,64,s128) s128x4x64;
typedef mat(4,128,s128) s128x4x128;

typedef mat(8,1,s128) s128x8x1;
typedef mat(8,2,s128) s128x8x2;
typedef mat(8,4,s128) s128x8x4;
typedef mat(8,8,s128) s128x8x8;
typedef mat(8,16,s128) s128x8x16;
typedef mat(8,32,s128) s128x8x32;
typedef mat(8,64,s128) s128x8x64;
typedef mat(8,128,s128) s128x8x128;

typedef mat(16,1,s128) s128x16x1;
typedef mat(16,2,s128) s128x16x2;
typedef mat(16,4,s128) s128x16x4;
typedef mat(16,8,s128) s128x16x8;
typedef mat(16,16,s128) s128x16x16;
typedef mat(16,32,s128) s128x16x32;
typedef mat(16,64,s128) s128x16x64;
typedef mat(16,128,s128) s128x16x128;

typedef mat(32,1,s128) s128x32x1;
typedef mat(32,2,s128) s128x32x2;
typedef mat(32,4,s128) s128x32x4;
typedef mat(32,8,s128) s128x32x8;
typedef mat(32,16,s128) s128x32x16;
typedef mat(32,32,s128) s128x32x32;
typedef mat(32,64,s128) s128x32x64;
typedef mat(32,128,s128) s128x32x128;

typedef mat(64,1,s128) s128x64x1;
typedef mat(64,2,s128) s128x64x2;
typedef mat(64,4,s128) s128x64x4;
typedef mat(64,8,s128) s128x64x8;
typedef mat(64,16,s128) s128x64x16;
typedef mat(64,32,s128) s128x64x32;
typedef mat(64,64,s128) s128x64x64;
typedef mat(64,128,s128) s128x64x128;

typedef mat(128,1,s128) s128x128x1;
typedef mat(128,2,s128) s128x128x2;
typedef mat(128,4,s128) s128x128x4;
typedef mat(128,8,s128) s128x128x8;
typedef mat(128,16,s128) s128x128x16;
typedef mat(128,32,s128) s128x128x32;
typedef mat(128,64,s128) s128x128x64;
typedef mat(128,128,s128) s128x128x128;


typedef mat(1,1,f16) f16x1x1;
typedef mat(1,2,f16) f16x1x2;
typedef mat(1,4,f16) f16x1x4;
typedef mat(1,8,f16) f16x1x8;
typedef mat(1,16,f16) f16x1x16;
typedef mat(1,32,f16) f16x1x32;
typedef mat(1,64,f16) f16x1x64;
typedef mat(1,128,f16) f16x1x128;

typedef mat(2,1,f16) f16x2x1;
typedef mat(2,2,f16) f16x2x2;
typedef mat(2,4,f16) f16x2x4;
typedef mat(2,8,f16) f16x2x8;
typedef mat(2,16,f16) f16x2x16;
typedef mat(2,32,f16) f16x2x32;
typedef mat(2,64,f16) f16x2x64;
typedef mat(2,128,f16) f16x2x128;

typedef mat(4,1,f16) f16x4x1;
typedef mat(4,2,f16) f16x4x2;
typedef mat(4,4,f16) f16x4x4;
typedef mat(4,8,f16) f16x4x8;
typedef mat(4,16,f16) f16x4x16;
typedef mat(4,32,f16) f16x4x32;
typedef mat(4,64,f16) f16x4x64;
typedef mat(4,128,f16) f16x4x128;

typedef mat(8,1,f16) f16x8x1;
typedef mat(8,2,f16) f16x8x2;
typedef mat(8,4,f16) f16x8x4;
typedef mat(8,8,f16) f16x8x8;
typedef mat(8,16,f16) f16x8x16;
typedef mat(8,32,f16) f16x8x32;
typedef mat(8,64,f16) f16x8x64;
typedef mat(8,128,f16) f16x8x128;

typedef mat(16,1,f16) f16x16x1;
typedef mat(16,2,f16) f16x16x2;
typedef mat(16,4,f16) f16x16x4;
typedef mat(16,8,f16) f16x16x8;
typedef mat(16,16,f16) f16x16x16;
typedef mat(16,32,f16) f16x16x32;
typedef mat(16,64,f16) f16x16x64;
typedef mat(16,128,f16) f16x16x128;

typedef mat(32,1,f16) f16x32x1;
typedef mat(32,2,f16) f16x32x2;
typedef mat(32,4,f16) f16x32x4;
typedef mat(32,8,f16) f16x32x8;
typedef mat(32,16,f16) f16x32x16;
typedef mat(32,32,f16) f16x32x32;
typedef mat(32,64,f16) f16x32x64;
typedef mat(32,128,f16) f16x32x128;

typedef mat(64,1,f16) f16x64x1;
typedef mat(64,2,f16) f16x64x2;
typedef mat(64,4,f16) f16x64x4;
typedef mat(64,8,f16) f16x64x8;
typedef mat(64,16,f16) f16x64x16;
typedef mat(64,32,f16) f16x64x32;
typedef mat(64,64,f16) f16x64x64;
typedef mat(64,128,f16) f16x64x128;

typedef mat(128,1,f16) f16x128x1;
typedef mat(128,2,f16) f16x128x2;
typedef mat(128,4,f16) f16x128x4;
typedef mat(128,8,f16) f16x128x8;
typedef mat(128,16,f16) f16x128x16;
typedef mat(128,32,f16) f16x128x32;
typedef mat(128,64,f16) f16x128x64;
typedef mat(128,128,f16) f16x128x128;


typedef mat(1,1,f32) f32x1x1;
typedef mat(1,2,f32) f32x1x2;
typedef mat(1,4,f32) f32x1x4;
typedef mat(1,8,f32) f32x1x8;
typedef mat(1,16,f32) f32x1x16;
typedef mat(1,32,f32) f32x1x32;
typedef mat(1,64,f32) f32x1x64;
typedef mat(1,128,f32) f32x1x128;

typedef mat(2,1,f32) f32x2x1;
typedef mat(2,2,f32) f32x2x2;
typedef mat(2,4,f32) f32x2x4;
typedef mat(2,8,f32) f32x2x8;
typedef mat(2,16,f32) f32x2x16;
typedef mat(2,32,f32) f32x2x32;
typedef mat(2,64,f32) f32x2x64;
typedef mat(2,128,f32) f32x2x128;

typedef mat(4,1,f32) f32x4x1;
typedef mat(4,2,f32) f32x4x2;
typedef mat(4,4,f32) f32x4x4;
typedef mat(4,8,f32) f32x4x8;
typedef mat(4,16,f32) f32x4x16;
typedef mat(4,32,f32) f32x4x32;
typedef mat(4,64,f32) f32x4x64;
typedef mat(4,128,f32) f32x4x128;

typedef mat(8,1,f32) f32x8x1;
typedef mat(8,2,f32) f32x8x2;
typedef mat(8,4,f32) f32x8x4;
typedef mat(8,8,f32) f32x8x8;
typedef mat(8,16,f32) f32x8x16;
typedef mat(8,32,f32) f32x8x32;
typedef mat(8,64,f32) f32x8x64;
typedef mat(8,128,f32) f32x8x128;

typedef mat(16,1,f32) f32x16x1;
typedef mat(16,2,f32) f32x16x2;
typedef mat(16,4,f32) f32x16x4;
typedef mat(16,8,f32) f32x16x8;
typedef mat(16,16,f32) f32x16x16;
typedef mat(16,32,f32) f32x16x32;
typedef mat(16,64,f32) f32x16x64;
typedef mat(16,128,f32) f32x16x128;

typedef mat(32,1,f32) f32x32x1;
typedef mat(32,2,f32) f32x32x2;
typedef mat(32,4,f32) f32x32x4;
typedef mat(32,8,f32) f32x32x8;
typedef mat(32,16,f32) f32x32x16;
typedef mat(32,32,f32) f32x32x32;
typedef mat(32,64,f32) f32x32x64;
typedef mat(32,128,f32) f32x32x128;

typedef mat(64,1,f32) f32x64x1;
typedef mat(64,2,f32) f32x64x2;
typedef mat(64,4,f32) f32x64x4;
typedef mat(64,8,f32) f32x64x8;
typedef mat(64,16,f32) f32x64x16;
typedef mat(64,32,f32) f32x64x32;
typedef mat(64,64,f32) f32x64x64;
typedef mat(64,128,f32) f32x64x128;

typedef mat(128,1,f32) f32x128x1;
typedef mat(128,2,f32) f32x128x2;
typedef mat(128,4,f32) f32x128x4;
typedef mat(128,8,f32) f32x128x8;
typedef mat(128,16,f32) f32x128x16;
typedef mat(128,32,f32) f32x128x32;
typedef mat(128,64,f32) f32x128x64;
typedef mat(128,128,f32) f32x128x128;


typedef mat(1,1,f64) f64x1x1;
typedef mat(1,2,f64) f64x1x2;
typedef mat(1,4,f64) f64x1x4;
typedef mat(1,8,f64) f64x1x8;
typedef mat(1,16,f64) f64x1x16;
typedef mat(1,32,f64) f64x1x32;
typedef mat(1,64,f64) f64x1x64;
typedef mat(1,128,f64) f64x1x128;

typedef mat(2,1,f64) f64x2x1;
typedef mat(2,2,f64) f64x2x2;
typedef mat(2,4,f64) f64x2x4;
typedef mat(2,8,f64) f64x2x8;
typedef mat(2,16,f64) f64x2x16;
typedef mat(2,32,f64) f64x2x32;
typedef mat(2,64,f64) f64x2x64;
typedef mat(2,128,f64) f64x2x128;

typedef mat(4,1,f64) f64x4x1;
typedef mat(4,2,f64) f64x4x2;
typedef mat(4,4,f64) f64x4x4;
typedef mat(4,8,f64) f64x4x8;
typedef mat(4,16,f64) f64x4x16;
typedef mat(4,32,f64) f64x4x32;
typedef mat(4,64,f64) f64x4x64;
typedef mat(4,128,f64) f64x4x128;

typedef mat(8,1,f64) f64x8x1;
typedef mat(8,2,f64) f64x8x2;
typedef mat(8,4,f64) f64x8x4;
typedef mat(8,8,f64) f64x8x8;
typedef mat(8,16,f64) f64x8x16;
typedef mat(8,32,f64) f64x8x32;
typedef mat(8,64,f64) f64x8x64;
typedef mat(8,128,f64) f64x8x128;

typedef mat(16,1,f64) f64x16x1;
typedef mat(16,2,f64) f64x16x2;
typedef mat(16,4,f64) f64x16x4;
typedef mat(16,8,f64) f64x16x8;
typedef mat(16,16,f64) f64x16x16;
typedef mat(16,32,f64) f64x16x32;
typedef mat(16,64,f64) f64x16x64;
typedef mat(16,128,f64) f64x16x128;

typedef mat(32,1,f64) f64x32x1;
typedef mat(32,2,f64) f64x32x2;
typedef mat(32,4,f64) f64x32x4;
typedef mat(32,8,f64) f64x32x8;
typedef mat(32,16,f64) f64x32x16;
typedef mat(32,32,f64) f64x32x32;
typedef mat(32,64,f64) f64x32x64;
typedef mat(32,128,f64) f64x32x128;

typedef mat(64,1,f64) f64x64x1;
typedef mat(64,2,f64) f64x64x2;
typedef mat(64,4,f64) f64x64x4;
typedef mat(64,8,f64) f64x64x8;
typedef mat(64,16,f64) f64x64x16;
typedef mat(64,32,f64) f64x64x32;
typedef mat(64,64,f64) f64x64x64;
typedef mat(64,128,f64) f64x64x128;

typedef mat(128,1,f64) f64x128x1;
typedef mat(128,2,f64) f64x128x2;
typedef mat(128,4,f64) f64x128x4;
typedef mat(128,8,f64) f64x128x8;
typedef mat(128,16,f64) f64x128x16;
typedef mat(128,32,f64) f64x128x32;
typedef mat(128,64,f64) f64x128x64;
typedef mat(128,128,f64) f64x128x128;

#undef vec
#undef mat

// some base types
// base types are optimized to be <= 16 bytes since 16 byte structs generally pass by value in 2 host GPRs
// whereas > 16 bytes are either memcopied or use simd registers
//
// functions that modify a base type return a new base type. functions that dont do not
// functions that both modify a base type _and_ return something else are not allowed
// e.g.
//  vector delete_by_index(vector_u32, idx) -> allowed
//  u32    at(vector_u32,idx) -> allowed
//  u32,vector pop(vector_u32, idx) -> not allowed, as pop modifies the vector _and_ gives a value


// arr - array - array of T with a size known at compile time
// vec - vector - array of T with a size known at runtime
// str - string - array of char with a size known at compile time with a '\0' terminator
// dynstr - dynamic string - array of char with a size known at run time with a '\0' terminator

#define ARR(T, N, name) \
	typedef struct name { T* arr; } name; \
	public name name##_new(void) {name ret; ret.arr = malloc(sizeof(T) * N); return ret; } \
	public name name##_new(T inp[static N]) {name ret; ret.arr = malloc(sizeof(T) * N); for(int i = 0; i < N; i++) ret.arr[i] = inp[i]; return ret; } \
	public s32 len(name name){return N;} \
	public T at(name name, s32 i){ return name.arr[i];} \
  public name del(name name) { free(name.arr); name.arr = NULL; return name;}

#define VEC(T, name) \
	typedef struct name {T* vec; s32 len; s32 cap;} name; \
	public name name##_new(void) {name ret; ret.len = 0; ret.cap = 8; ret.vec = malloc(sizeof(T) * ret.cap); return ret; } \
	public name name##_new(s32 n, T inp[n]) {name ret; ret.len = n; ret.cap = (n*3)/2; ret.vec = malloc(sizeof(T) * ret.cap); for(int i = 0; i < n; i++) ret.vec[i] = inp[i]; return ret; }\
	public s32 len(name vec){ return vec.len;}\
	public s32 size(name vec){ return vec.cap * sizeof(T);}\
	public T at(name vec, s32 i){return vec.vec[i];}\
	public name del(name vec){free(vec.vec); vec.vec = NULL; return vec;} \
	public name del_idx(name vec, s32 idx) {for(int i = idx; i < vec.len - 1; i++) vec.vec[i] = vec.vec[i+1]; vec.len--; return vec;} \
	public name insert(name vec, s32 idx, T data){if(vec.len == vec.cap){vec.cap = vec.cap*3/2; vec.vec = realloc(vec.vec, vec.cap*sizeof(T));} vec.len++; for(int i = vec.len; i >= idx; i--){vec.vec[i] = vec.vec[i-1];} return vec;}



VEC(u8, u8vec)


//lots o function wrappers around operators
#define OPERATOR_WRAPPER_INT(op, name)\
	public constfunc u8 name(u8 a, u8 b){return a op b;}\
	public constfunc s8 name(s8 a, s8 b){return a op b;}\
	public constfunc u16 name(u16 a, u16 b){return a op b;}\
	public constfunc s16 name(s16 a, s16 b){return a op b;}\
	public constfunc u32 name(u32 a, u32 b){return a op b;}\
	public constfunc s32 name(s32 a, s32 b){return a op b;}\
	public constfunc u64 name(u64 a, u64 b){return a op b;}\
	public constfunc s64 name(s64 a, s64 b){return a op b;}\
	public constfunc u128 name(u128 a, u128 b){return a op b;}\
	public constfunc s128 name(s128 a, s128 b){return a op b;}

#define OPERATOR_WRAPPER_FLOAT(op, name)\
	public constfunc f16 name(f16 a, f16 b){return a op b;}\
	public constfunc f32 name(f32 a, f32 b){return a op b;}\
	public constfunc f64 name(f64 a, f64 b){return a op b;}

OPERATOR_WRAPPER_INT(+, add)
OPERATOR_WRAPPER_INT(-, sub)
OPERATOR_WRAPPER_INT(*, mul)
OPERATOR_WRAPPER_INT(/, div)
OPERATOR_WRAPPER_INT(>, gt)
OPERATOR_WRAPPER_INT(>=, ge)
OPERATOR_WRAPPER_INT(<, lt)
OPERATOR_WRAPPER_INT(<=, le)
OPERATOR_WRAPPER_INT(==, eq)
OPERATOR_WRAPPER_INT(!=, neq)
OPERATOR_WRAPPER_INT(%, mod)
OPERATOR_WRAPPER_INT(&, and)
OPERATOR_WRAPPER_INT(|, or)
OPERATOR_WRAPPER_INT(^, xor)
OPERATOR_WRAPPER_INT(<<, lsl)

OPERATOR_WRAPPER_FLOAT(+, add)
OPERATOR_WRAPPER_FLOAT(-, sub)
OPERATOR_WRAPPER_FLOAT(*, mul)
OPERATOR_WRAPPER_FLOAT(/, div)
OPERATOR_WRAPPER_FLOAT(>, gt)
OPERATOR_WRAPPER_FLOAT(>=, ge)
OPERATOR_WRAPPER_FLOAT(<, lt)
OPERATOR_WRAPPER_FLOAT(<=, le)
OPERATOR_WRAPPER_FLOAT(==, eq)
OPERATOR_WRAPPER_FLOAT(!=, neq)

public constfunc f16 mod(f16 a, f16 b){unreachable();}
public constfunc f32 mod(f32 a, f32 b){unreachable();}
public constfunc f64 mod(f64 a, f64 b){unreachable();}

public constfunc f16 and(f16 a, f16 b){unreachable();}
public constfunc f32 and(f32 a, f32 b){unreachable();}
public constfunc f64 and(f64 a, f64 b){unreachable();}

public constfunc f16 or(f16 a, f16 b){unreachable();}
public constfunc f32 or(f32 a, f32 b){unreachable();}
public constfunc f64 or(f64 a, f64 b){unreachable();}

public constfunc f16 xor(f16 a, f16 b){unreachable();}
public constfunc f32 xor(f32 a, f32 b){unreachable();}
public constfunc f64 xor(f64 a, f64 b){unreachable();}

public constfunc f16 lsl(f16 a, f16 b){unreachable();}
public constfunc f32 lsl(f32 a, f32 b){unreachable();}
public constfunc f64 lsl(f64 a, f64 b){unreachable();}

public constfunc f16 lsr(f16 a, f16 b){unreachable();}
public constfunc f32 lsr(f32 a, f32 b){unreachable();}
public constfunc f64 lsr(f64 a, f64 b){unreachable();}

public constfunc f16 asr(f16 a, f16 b){unreachable();}
public constfunc f32 asr(f32 a, f32 b){unreachable();}
public constfunc f64 asr(f64 a, f64 b){unreachable();}


public constfunc s8   lsr(s8 a, s8 b){return (u8)a >> b ;}
public constfunc u8   lsr(u8 a, u8 b){return a >> b ;}
public constfunc s16  lsr(s16 a, s16 b){return (u16)a >> b ;}
public constfunc u16  lsr(u16 a, u16 b){return a >> b ;}
public constfunc s32  lsr(s32 a, s32 b){return (u32)a >> b ;}
public constfunc u32  lsr(u32 a, u32 b){return a >> b ;}
public constfunc s64  lsr(s64 a, s64 b){return (u64)a >> b ;}
public constfunc u64  lsr(u64 a, u64 b){return a >> b ;}
public constfunc s128 lsr(s128 a, s128 b){return (u128)a >> b ;}
public constfunc u128 lsr(u128 a, u128 b){return a >> b ;}


public constfunc s8   asr(s8 a, s8 b){return a >> b ;}
public constfunc u8   asr(u8 a, u8 b){return (s8)a >> b ;}
public constfunc s16  asr(s16 a, s16 b){return a >> b ;}
public constfunc u16  asr(u16 a, u16 b){return (s16)a >> b ;}
public constfunc s32  asr(s32 a, s32 b){return a >> b ;}
public constfunc u32  asr(u32 a, u32 b){return (s32)a >> b ;}
public constfunc s64  asr(s64 a, s64 b){return a >> b ;}
public constfunc u64  asr(u64 a, u64 b){return (s64)a >> b ;}
public constfunc s128 asr(s128 a, s128 b){return a >> b ;}
public constfunc u128 asr(u128 a, u128 b){return (s128)a >> b ;}


public constfunc s8   mac(s8 a, s8 b, s8 c){return a + b * c;}
public constfunc u8   mac(u8 a, u8 b, s8 c){return a + b * c;}
public constfunc s16  mac(s16 a, s16 b, s8 c){return a + b * c;}
public constfunc u16  mac(u16 a, u16 b, s8 c){return a + b * c;}
public constfunc s32  mac(s32 a, s32 b, s8 c){return a + b * c;}
public constfunc u32  mac(u32 a, u32 b, s8 c){return a + b * c;}
public constfunc s64  mac(s64 a, s64 b, s8 c){return a + b * c;}
public constfunc u64  mac(u64 a, u64 b, s8 c){return a + b * c;}
public constfunc s128 mac(s128 a, s128 b, s8 c){return a + b * c;}
public constfunc u128 mac(u128 a, u128 b, s8 c){return a + b * c;}
public constfunc f16  mac(f16 a, f16 b, s8 c){return a + b * c;}
public constfunc f32  mac(f32 a, f32 b, s8 c){return a + b * c;}
public constfunc f64  mac(f64 a, f64 b, s8 c){return a + b * c;}

public constfunc s8   select(s8 cond, s8 a, s8 b){return cond ? a : b;}
public constfunc u8   select(u8 cond, u8 a, s8 b){return cond ? a : b;}
public constfunc s16  select(s16 cond, s16 a, s8 b){return cond ? a : b;}
public constfunc u16  select(u16 cond, u16 a, s8 b){return cond ? a : b;}
public constfunc s32  select(s32 cond, s32 a, s8 b){return cond ? a : b;}
public constfunc u32  select(u32 cond, u32 a, s8 b){return cond ? a : b;}
public constfunc s64  select(s64 cond, s64 a, s8 b){return cond ? a : b;}
public constfunc u64  select(u64 cond, u64 a, s8 b){return cond ? a : b;}
public constfunc s128 select(s128 cond, s128 a, s8 b){return cond ? a : b;}
public constfunc u128 select(u128 cond, u128 a, s8 b){return cond ? a : b;}
public constfunc f16  select(f16 cond, f16 a, s8 b){return cond ? a : b;}
public constfunc f32  select(f32 cond, f32 a, s8 b){return cond ? a : b;}
public constfunc f64  select(f64 cond, f64 a, s8 b){return cond ? a : b;}

