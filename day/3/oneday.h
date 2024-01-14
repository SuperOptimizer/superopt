#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

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
