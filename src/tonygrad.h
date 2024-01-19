#pragma once
#include "common.h"

typedef struct dims {
  s32 rank;
  union {
    s32 dims[8];
    s32 x,y,z,w,i,j,k,l;
  };
} dims;

typedef struct tensor {
  dtype dtype;
  dims dims;
  void* data;
} tensor;

public s32 size(dtype dtype) {
  switch(dtype) {
    case DT_S8: case DT_U8: return 1;
    case DT_S16: case DT_U16: case DT_F16: return 2;
    case DT_S32: case DT_U32: case DT_F32: return 4;
    case DT_S64: case DT_U64: case DT_F64: return 8;
    case DT_S128: case DT_U128:  return 16;
    default:
      unreachable();
  }
  unreachable();
  return -1;
}

public s32 len(tensor t) {
  s32 len = 1;
  for(int i = 0; i < t.dims.rank; i++) len *= t.dims.dims[i];
  return len;
}

public dims dims_new(s32 x){ return(dims) {1, {x,1,1,1,1,1,1,1,}}; }
public dims dims_new(s32 x, s32 y){ return(dims) {2,{x,y,1,1,1,1,1,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z){ return(dims) {3,{x,y,z,1,1,1,1,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z, s32 w){ return(dims) {4,{x,y,z,w,1,1,1,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z, s32 w, s32 i){ return(dims) {5,{x,y,z,w,i,1,1,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z, s32 w, s32 i, s32 j){ return(dims) {6,{x,y,z,w,i,j,1,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k){ return(dims) {7,{x,y,z,w,i,j,k,1,}}; }
public dims dims_new(s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, s32 l){ return(dims) {8,{x,y,z,w,i,j,k,l,}}; }

public tensor tensor_new(dtype dtype, dims dims) {
  tensor ret;
  ret.dtype = dtype;
  ret.dims = dims;
  s32 len = 1;
  for(int i = 0; i < dims.rank; i++) len *= ret.dims.dims[i];
  ret.data = malloc(len * size(dtype));
  return ret;
}

//TODO: can these at functions be const? they don't modify the pointed to data so maybe

public void* at(tensor t, dims idx) {
  assert(idx.rank == t.dims.rank);

  s32 offset = 0;
  s32 multiplier = 1;
  for (int i = t.dims.rank - 1; i >= 0; --i) {
    //if (idx.dims[i] < 0 || idx.dims[i] >= t.dims.dims[i]) {
    //  // Handle error: Index out of bounds
    //  return NULL;
    //}
    offset += idx.dims[i] * multiplier;
    multiplier *= t.dims.dims[i];
  }

  char* data = t.data;
  return data + offset * size(t.dtype);
}

public void* at(tensor t, s32 i0) {
  char* data = (char*)t.data;
  return (void*)(data + i0 * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1) {
  char* data = t.data;
  s32 offset = i0 * t.dims.dims[1]
             + i1;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2]
             + i1 * t.dims.dims[2]
             + i2;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2, s32 i3) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2] * t.dims.dims[3]
             + i1 * t.dims.dims[2] * t.dims.dims[3]
             + i2 * t.dims.dims[3]
             + i3;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2, s32 i3, s32 i4) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4]
             + i1 * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4]
             + i2 * t.dims.dims[3] * t.dims.dims[4]
             + i3 * t.dims.dims[4]
             + i4;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2, s32 i3, s32 i4, s32 i5) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5]
             + i1 * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5]
             + i2 * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5]
             + i3 * t.dims.dims[4] * t.dims.dims[5]
             + i4 * t.dims.dims[5]
             + i5;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2, s32 i3, s32 i4, s32 i5, s32 i6) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6]
             + i1 * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6]
             + i2 * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6]
             + i3 * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6]
             + i4 * t.dims.dims[5] * t.dims.dims[6]
             + i5 * t.dims.dims[6]
             + i6;
  return (void*)(data + offset * size(t.dtype));
}

public void* at(tensor t, s32 i0, s32 i1, s32 i2, s32 i3, s32 i4, s32 i5, s32 i6, s32 i7) {
  char* data = (char*)t.data;
  s32 offset = i0 * t.dims.dims[1] * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6] * t.dims.dims[7]
             + i1 * t.dims.dims[2] * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6] * t.dims.dims[7]
             + i2 * t.dims.dims[3] * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6] * t.dims.dims[7]
             + i3 * t.dims.dims[4] * t.dims.dims[5] * t.dims.dims[6] * t.dims.dims[7]
             + i4 * t.dims.dims[5] * t.dims.dims[6] * t.dims.dims[7]
             + i5 * t.dims.dims[6] * t.dims.dims[7]
             + i6 * t.dims.dims[7]
             + i7;
  return (void*)(data + offset * size(t.dtype));
}

#define GET_DEF(T, DT) \
public purefunc T get_##T(tensor t, dims dims){assert(t.dtype == DT); return *(T*)at(t,dims);}\
public purefunc T get_##T(tensor t, s32 x) {assert(t.dtype == DT); return *(T*)at(t,x);} \
public purefunc T get_##T(tensor t, s32 x, s32 y) {assert(t.dtype == DT); return *(T*)at(t,x,y);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z) {assert(t.dtype == DT); return *(T*)at(t,x,y,z);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z, s32 w) {assert(t.dtype == DT); return *(T*)at(t,x,y,z,w);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i) {assert(t.dtype == DT); return *(T*)at(t,x,y,z,w,i);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j) {assert(t.dtype == DT); return *(T*)at(t,x,y,z,w,i,j);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k) {assert(t.dtype == DT); return *(T*)at(t,x,y,z,w,i,j,k);} \
public purefunc T get_##T(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, s32 l) {assert(t.dtype == DT); return *(T*)at(t,x,y,z,w,i,j,k,l);} \


GET_DEF(u8, DT_U8)
GET_DEF(s8, DT_S8)
GET_DEF(u16, DT_U16)
GET_DEF(s16, DT_S16)
GET_DEF(u32, DT_U32)
GET_DEF(s32, DT_S32)
GET_DEF(u64, DT_U64)
GET_DEF(s64, DT_S64)
GET_DEF(u128, DT_U128)
GET_DEF(s128, DT_S128)
GET_DEF(f16, DT_F16)
GET_DEF(f32, DT_F32)
GET_DEF(f64, DT_F64)

#define SET_DEF(T, DT) \
  public purefunc void set(tensor t, dims dims, T data){assert(t.dtype == DT); *(T*)at(t,dims) = data;}\
  public purefunc void set(tensor t, s32 x, T data) {assert(t.dtype == DT); *(T*)at(t,x) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, T data) {assert(t.dtype == DT); *(T*)at(t,x,y) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, s32 w, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z,w) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z,w,i) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z,w,i,j) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z,w,i,j,k) = data;} \
  public purefunc void set(tensor t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, s32 l, T data) {assert(t.dtype == DT); *(T*)at(t,x,y,z,w,i,j,k,l) = data;} \

SET_DEF(u8, DT_U8)
SET_DEF(s8, DT_S8)
SET_DEF(u16, DT_U16)
SET_DEF(s16, DT_S16)
SET_DEF(u32, DT_U32)
SET_DEF(s32, DT_S32)
SET_DEF(u64, DT_U64)
SET_DEF(s64, DT_S64)
SET_DEF(u128, DT_U128)
SET_DEF(s128, DT_S128)
SET_DEF(f16, DT_F16)
SET_DEF(f32, DT_F32)
SET_DEF(f64, DT_F64)

#undef SET_DEF

#define SCALAR_DEF(name, op) \
  public constfunc s8 name(s8 a, s8 b){return a op b;}\
  public constfunc u8 name(u8 a, u8 b){return a op b;}\
  public constfunc s16 name(s16 a, s16 b){return a op b;}\
  public constfunc u16 name(u16 a, u16 b){return a op b;}\
  public constfunc s32 name(s32 a, s32 b){return a op b;}\
  public constfunc u32 name(u32 a, u32 b){return a op b;}\
  public constfunc s64 name(s64 a, s64 b){return a op b;}\
  public constfunc u64 name(u64 a, u64 b){return a op b;}\
  public constfunc u128 name(u128 a, u128 b){return a op b;}\
  public constfunc s128 name(s128 a, s128 b){return a op b;}\
  public constfunc f16 name(f16 a, f16 b){return a op b;}\
  public constfunc f32 name(f32 a, f32 b){return a op b;}\
  public constfunc f64 name(f64 a, f64 b){return a op b;}

SCALAR_DEF(add, +)
SCALAR_DEF(sub, -)
SCALAR_DEF(mul, *)
//SCALAR_DEF(div, /)

#undef SCALAR_DEF

#define for1(extent, body){for(int x = 0; x < extent.x; x++){body}
#define for2(extent, body)(for1(extent, for(int y = 0; y < extent.y; y++){body})
#define for3(extent, body)(for2(extent, for(int z = 0; z < extent.z; z++){body})
#define for4(extent, body)(for3(extent, for(int w = 0; w < extent.w; w++){body})
#define for5(extent, body)(for4(extent, for(int i = 0; i < extent.z; i++){body})
#define for6(extent, body)(for5(extent, for(int j = 0; j < extent.z; j++){body})
#define for7(extent, body)(for6(extent, for(int k = 0; k < extent.z; k++){body})
#define for8(extent, body)(for7(extent, for(int l = 0; l < extent.z; l++){body})

public tensor add(tensor a, tensor b) {
  assert(a.dims.rank == b.dims.rank);
  assert(memcmp(&a.dims,&b.dims,sizeof(dims)));
  assert(a.dtype == b.dtype);
  tensor ret = tensor_new(a.dtype, a.dims);
  switch(a.dims.rank) {
    case 1: {
      switch(a.dtype) {
        case DT_U8: for1(ret.dims, set(ret, x,))
      }
    }
  }
  for(int i  = 0; i < len(a); i++) {
    switch(a.dtype) {
      case DT_U8: *(u8*)&ret.data[i] = add(*(u8*)&a.data[i], *(u8*)&b.data[i]); break;

    }

  }
}