#pragma once
#include "common.h"





typedef struct dims {
  s32 rank;
  union {
    s32 dims[8];
    s32 x,y,z,w,i,j,k,l;
  };
} dims;

typedef dims idx;

public dims dims_new(s32 rank, s32 dims_[static 8]) {
  dims ret;
  ret.rank = rank;
  for(int i=0;i<rank;i++)ret.dims[i] = dims_[i];
  return ret;
}

public idx idx_new(s32 rank, s32 dims_[static 8]) {
  s32 i[] = {0,0,0,0,0,0,0,0};
  return dims_new(rank, i);
}


#define DIMSDEF(_rank, ...) \
  typedef struct dims##_rank{\
  s32 rank; \
  union{\
  s32 dims[8];\
    s32 x,y,z,w,i,j,k,l;\
  };\
  } dims##_rank;\
  public dims##_rank dims##_rank##_new(s32 dims[static _rank]){ dims##_rank ret; ret.rank=_rank; for(int i=0;i<_rank;i++)ret.dims[i] = dims[i]; return ret;}


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

DIMSDEF(1,x)
DIMSDEF(2,x,y)
DIMSDEF(3,x,y,z)
DIMSDEF(4,x,y,z,w)
DIMSDEF(5,x,y,z,w,i)
DIMSDEF(6,x,y,z,w,i,j)
DIMSDEF(7,x,y,z,w,i,j,k)
DIMSDEF(8,x,y,z,w,i,j,k,l)


typedef struct tensor {
  dtype dtype;
  dims dims;
  void* restrict data;
} tensor;

#define TENSORDEF(_dtype, DT) \
  typedef struct tensor_##_dtype {\
    dtype dtype; \
    dims dims;\
    _dtype* restrict data;\
  } tensor_##_dtype;\
  public tensor_##_dtype tensor_##_dtype##_new(dims dims){\
  tensor_##_dtype ret;\
  ret.dtype = DT;\
  ret.dims = dims;\
  s32 len = 1;\
  for(int i = 0; i < dims.rank; i++) len *= ret.dims.dims[i];\
  ret.data = malloc(len * size(DT));\
  return ret; \
  }

TENSORDEF(u8,   DT_U8)
TENSORDEF(s8,   DT_S8)
TENSORDEF(u16,  DT_U16)
TENSORDEF(s16,  DT_S16)
TENSORDEF(u32,  DT_U32)
TENSORDEF(s32,  DT_S32)
TENSORDEF(u64,  DT_U64)
TENSORDEF(s64,  DT_S64)
TENSORDEF(u128, DT_U128)
TENSORDEF(s128, DT_S128)
TENSORDEF(f16,  DT_F16)
TENSORDEF(f32,  DT_F32)
TENSORDEF(f64,  DT_F64)


public bool eq(dims a, dims b) {
  if(a.rank != b.rank) return false;
  for(int i = 0; i < a.rank; i++) {
    if(a.dims[i] != b.dims[i]) {
      return false;
    }
  }
  return true;
}

public bool lt(dims idx, dims extent) {
  assert(idx.rank == extent.rank);
  for(int i = idx.rank -1; i >= 0; i--) {
    if(idx.dims[i] < extent.dims[i]-1) {
      return true;
    }
  }
  return false;
}


public s32 len(tensor t) {
  s32 len = 1;
  for(int i = 0; i < t.dims.rank; i++) len *= t.dims.dims[i];
  return len;
}


public dims inc(dims idx, dims extent) {
  for(int i = idx.rank - 1; i >= 0; i--) {
    idx.dims[i]++;
    if(idx.dims[i] < extent.dims[i]) {
      return idx;
    }
    if (idx.dims[i] == extent.dims[i]) {
      idx.dims[i] = 0;
    } else {
      unreachable();
    }
  }
  // trying to increment idx past the extent
  unreachable();
}

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

#undef GET_DEF
#undef SET_DEF

#define WRAP_GET(T) \
public purefunc T get(tensor_##T t, dims dims){ return get_##T(*(tensor*)&t, dims);}\
public purefunc T get(tensor_##T t, s32 x) { return get_##T(*(tensor*)&t, x);} \
public purefunc T get(tensor_##T t, s32 x, s32 y) { return get_##T(*(tensor*)&t, x,y);}  \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z) {return get_##T(*(tensor*)&t, x,y,z);} \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z, s32 w) {return get_##T(*(tensor*)&t, x,y,z,w);}  \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i) {return get_##T(*(tensor*)&t, x,y,z,w,i); } \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j) {return get_##T(*(tensor*)&t, x,y,z,w,i,j); } \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k) {return get_##T(*(tensor*)&t, x,y,z,w,i,j,k); } \
public purefunc T get(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, s32 l) {return get_##T(*(tensor*)&t, x,y,z,w,i,j,k,l); } \

#define WRAP_SET(T) \
public purefunc void set(tensor_##T t, dims dims, T data){set(*(tensor*)&t,dims, data);}\
public purefunc void set(tensor_##T t, s32 x, T data) {set(*(tensor*)&t,x, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, T data) {set(*(tensor*)&t,x,y, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, T data) {set(*(tensor*)&t,x,y,z, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, s32 w, T data) {set(*(tensor*)&t,x,y,z,w, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, T data) {set(*(tensor*)&t,x,y,z,w,i, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, T data) {set(*(tensor*)&t,x,y,z,w,i,j, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, T data) {set(*(tensor*)&t,x,y,z,w,i,j,k, data);} \
public purefunc void set(tensor_##T t, s32 x, s32 y, s32 z, s32 w, s32 i, s32 j, s32 k, s32 l, T data) {set(*(tensor*)&t,x,y,z,w,i,j,k,l, data);} \


WRAP_GET(u8)
WRAP_GET(s8)
WRAP_GET(u16)
WRAP_GET(s16)
WRAP_GET(u32)
WRAP_GET(s32)
WRAP_GET(u64)
WRAP_GET(s64)
WRAP_GET(u128)
WRAP_GET(s128)
WRAP_GET(f16)
WRAP_GET(f32)
WRAP_GET(f64)

WRAP_SET(u8)
WRAP_SET(s8)
WRAP_SET(u16)
WRAP_SET(s16)
WRAP_SET(u32)
WRAP_SET(s32)
WRAP_SET(u64)
WRAP_SET(s64)
WRAP_SET(u128)
WRAP_SET(s128)
WRAP_SET(f16)
WRAP_SET(f32)
WRAP_SET(f64)

#undef WRAP_GET
#undef WRAP_SET

#define for1(extent, body)for(int x = 0; x < extent.x; x++){body}
#define for2(extent, body)for1(extent, for(int y = 0; y < extent.y; y++){body}
#define for3(extent, body)for2(extent, for(int z = 0; z < extent.z; z++){body}
#define for4(extent, body)for3(extent, for(int w = 0; w < extent.w; w++){body}
#define for5(extent, body)for4(extent, for(int i = 0; i < extent.z; i++){body}
#define for6(extent, body)for5(extent, for(int j = 0; j < extent.z; j++){body}
#define for7(extent, body)for6(extent, for(int k = 0; k < extent.z; k++){body}
#define for8(extent, body)for7(extent, for(int l = 0; l < extent.z; l++){body}

#define SCALAR_TENSOR_DEFS(func) \
  public tensor func(tensor a, tensor b, dtype dtype) {\
    assert(a.dims.rank == b.dims.rank);\
    assert(a.dtype == b.dtype && b.dtype == dtype); \
    assert(eq(a.dims,b.dims));\
    tensor ret = tensor_new(a.dtype, a.dims);\
    for(idx i = idx_new(a.dims.rank,a.dims.dims); lt(i, a.dims); i = inc(i, a.dims)) {\
      switch(dtype) {\
        case DT_U8:   set(ret, i, func(get_u8(a,i),    get_u8(b,i)));  break;\
        case DT_S8:   set(ret, i, func(get_s8(a,i),    get_s8(b,i)));  break;\
        case DT_U16:  set(ret, i, func(get_u16(a,i),   get_u16(b,i)));  break;\
        case DT_S16:  set(ret, i, func(get_s16(a,i),   get_s16(b,i)));  break;\
        case DT_U32:  set(ret, i, func(get_u32(a,i),   get_u32(b,i)));  break;\
        case DT_S32:  set(ret, i, func(get_s32(a,i),   get_s32(b,i)));  break;\
        case DT_U64:  set(ret, i, func(get_u64(a,i),   get_u64(b,i)));  break;\
        case DT_S64:  set(ret, i, func(get_s64(a,i),   get_s64(b,i)));  break;\
        case DT_U128: set(ret, i, func(get_u128(a,i),  get_u128(b,i)));  break;\
        case DT_S128: set(ret, i, func(get_s128(a,i),  get_s128(b,i)));  break;\
        case DT_F16:  set(ret, i, func(get_f16(a,i),   get_f16(b,i)));  break;\
        case DT_F32:  set(ret, i, func(get_f32(a,i),   get_f32(b,i)));  break;\
        case DT_F64:  set(ret, i, func(get_f64(a,i),   get_f64(b,i)));  break;\
      }\
    }\
    return ret;\
  }

SCALAR_TENSOR_DEFS(add)
SCALAR_TENSOR_DEFS(sub)
SCALAR_TENSOR_DEFS(mul)
SCALAR_TENSOR_DEFS(div)
SCALAR_TENSOR_DEFS(gt)
SCALAR_TENSOR_DEFS(ge)
SCALAR_TENSOR_DEFS(lt)
SCALAR_TENSOR_DEFS(le)
SCALAR_TENSOR_DEFS(eq)
SCALAR_TENSOR_DEFS(neq)
SCALAR_TENSOR_DEFS(mod)
SCALAR_TENSOR_DEFS(and)
SCALAR_TENSOR_DEFS(or)
SCALAR_TENSOR_DEFS(xor)
SCALAR_TENSOR_DEFS(lsl)
SCALAR_TENSOR_DEFS(lsr)
SCALAR_TENSOR_DEFS(asr)


#define SCALAR_TENSOR_3_DEFS(func) \
  public tensor func(tensor a, tensor b, tensor c, dtype dtype) {\
  assert(a.dims.rank == b.dims.rank && b.dims.rank == c.dims.rank);\
  assert(a.dtype == b.dtype && b.dtype == c.dtype && c.dtype == dtype); \
  assert(eq(a.dims,b.dims) && eq(b.dims,c.dims));\
  tensor ret = tensor_new(a.dtype, a.dims);\
  for(idx i = idx_new(a.dims.rank,a.dims.dims); lt(i, a.dims); i = inc(i, a.dims)) {\
    switch(dtype) {\
      case DT_U8:   set(ret, i, func(get_u8(a,i),    get_u8(b,i),   get_u8(c,i)));  break;\
      case DT_S8:   set(ret, i, func(get_s8(a,i),    get_s8(b,i),   get_s8(c,i)));  break;\
      case DT_U16:  set(ret, i, func(get_u16(a,i),   get_u16(b,i),  get_u16(c,i)));  break;\
      case DT_S16:  set(ret, i, func(get_s16(a,i),   get_s16(b,i),  get_s16(c,i)));  break;\
      case DT_U32:  set(ret, i, func(get_u32(a,i),   get_u32(b,i),  get_u32(c,i)));  break;\
      case DT_S32:  set(ret, i, func(get_s32(a,i),   get_s32(b,i),  get_s32(c,i)));  break;\
      case DT_U64:  set(ret, i, func(get_u64(a,i),   get_u64(b,i),  get_u64(c,i)));  break;\
      case DT_S64:  set(ret, i, func(get_s64(a,i),   get_s64(b,i),  get_s64(c,i)));  break;\
      case DT_U128: set(ret, i, func(get_u128(a,i),  get_u128(b,i), get_u128(c,i)));  break;\
      case DT_S128: set(ret, i, func(get_s128(a,i),  get_s128(b,i), get_s128(c,i)));  break;\
      case DT_F16:  set(ret, i, func(get_f16(a,i),   get_f16(b,i),  get_f16(c,i)));  break;\
      case DT_F32:  set(ret, i, func(get_f32(a,i),   get_f32(b,i),  get_f32(c,i)));  break;\
      case DT_F64:  set(ret, i, func(get_f64(a,i),   get_f64(b,i),  get_f64(c,i)));  break;\
    }\
  }\
  return ret;\
  }

SCALAR_TENSOR_3_DEFS(select)
SCALAR_TENSOR_3_DEFS(mac)

#define SCALAR_TENSOR_TYPE_WRAP(func)\
public tensor_u8   func(tensor_u8 a,   tensor_u8 b)   {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_u8*)&ret;}\
public tensor_s8   func(tensor_s8 a,   tensor_s8 b)   {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_s8*)&ret;}\
public tensor_u16  func(tensor_u16 a,  tensor_u16 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_u16*)&ret;}\
public tensor_s16  func(tensor_s16 a,  tensor_s16 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_s16*)&ret;}\
public tensor_u32  func(tensor_u32 a,  tensor_u32 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_u32*)&ret;}\
public tensor_s32  func(tensor_s32 a,  tensor_s32 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_s32*)&ret;}\
public tensor_u64  func(tensor_u64 a,  tensor_u64 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_u64*)&ret;}\
public tensor_s64  func(tensor_s64 a,  tensor_s64 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_s64*)&ret;}\
public tensor_u128 func(tensor_u128 a, tensor_u128 b) {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_u128*)&ret;}\
public tensor_s128 func(tensor_s128 a, tensor_s128 b) {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_s128*)&ret;}\
public tensor_f16  func(tensor_f16 a,  tensor_f16 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_f16*)&ret;}\
public tensor_f32  func(tensor_f32 a,  tensor_f32 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_f32*)&ret;}\
public tensor_f64  func(tensor_f64 a,  tensor_f64 b)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, a.dtype); return *(tensor_f64*)&ret;}


#define SCALAR_TENSOR_3_TYPE_WRAP(func)\
public tensor_u8   func(tensor_u8 a,   tensor_u8 b,   tensor_u8 c)   {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_u8*)&ret;}\
public tensor_s8   func(tensor_s8 a,   tensor_s8 b,   tensor_s8 c)   {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_s8*)&ret;}\
public tensor_u16  func(tensor_u16 a,  tensor_u16 b,  tensor_u16 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_u16*)&ret;}\
public tensor_s16  func(tensor_s16 a,  tensor_s16 b,  tensor_s16 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_s16*)&ret;}\
public tensor_u32  func(tensor_u32 a,  tensor_u32 b,  tensor_u32 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_u32*)&ret;}\
public tensor_s32  func(tensor_s32 a,  tensor_s32 b,  tensor_s32 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_s32*)&ret;}\
public tensor_u64  func(tensor_u64 a,  tensor_u64 b,  tensor_u64 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_u64*)&ret;}\
public tensor_s64  func(tensor_s64 a,  tensor_s64 b,  tensor_s64 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_s64*)&ret;}\
public tensor_u128 func(tensor_u128 a, tensor_u128 b, tensor_u128 c) {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_u128*)&ret;}\
public tensor_s128 func(tensor_s128 a, tensor_s128 b, tensor_s128 c) {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_s128*)&ret;}\
public tensor_f16  func(tensor_f16 a,  tensor_f16 b,  tensor_f16 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_f16*)&ret;}\
public tensor_f32  func(tensor_f32 a,  tensor_f32 b,  tensor_f32 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_f32*)&ret;}\
public tensor_f64  func(tensor_f64 a,  tensor_f64 b,  tensor_f64 c)  {auto ret = func(*(tensor*)&a, *(tensor*)&b, *(tensor*)&c, a.dtype); return *(tensor_f64*)&ret;}

SCALAR_TENSOR_TYPE_WRAP(add)
SCALAR_TENSOR_TYPE_WRAP(sub)
SCALAR_TENSOR_TYPE_WRAP(mul)
SCALAR_TENSOR_TYPE_WRAP(div)
SCALAR_TENSOR_TYPE_WRAP(gt)
SCALAR_TENSOR_TYPE_WRAP(ge)
SCALAR_TENSOR_TYPE_WRAP(lt)
SCALAR_TENSOR_TYPE_WRAP(le)
SCALAR_TENSOR_TYPE_WRAP(eq)
SCALAR_TENSOR_TYPE_WRAP(neq)
SCALAR_TENSOR_TYPE_WRAP(mod)
SCALAR_TENSOR_TYPE_WRAP(and)
SCALAR_TENSOR_TYPE_WRAP(or)
SCALAR_TENSOR_TYPE_WRAP(xor)
SCALAR_TENSOR_TYPE_WRAP(lsl)
SCALAR_TENSOR_TYPE_WRAP(lsr)
SCALAR_TENSOR_TYPE_WRAP(asr)

SCALAR_TENSOR_3_TYPE_WRAP(select)
SCALAR_TENSOR_3_TYPE_WRAP(mac)

//activations

OPERATOR_WRAPPER_INT(*, linear)
OPERATOR_WRAPPER_FLOAT(*, linear)

//TODO: elu should do e^z not 3^z but 3 is kinda close so
public constfunc u8   elu(u8 z,   u8 alpha)  {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc s8   elu(s8 z,   s8 alpha)  {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc u16  elu(u16 z,  u16 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc s16  elu(s16 z,  s16 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc u32  elu(u32 z,  u32 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc s32  elu(s32 z,  s32 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc u64  elu(u64 z,  u64 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc s64  elu(s64 z,  s64 alpha) {return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc u128 elu(u128 z, u128 alpha){return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc s128 elu(s128 z, s128 alpha){return z > 0 ? z : alpha*(pow(3,z)-1);}
public constfunc f16  elu(f16 z,  f16 alpha) {return z > 0.0f ? z : alpha*(pow(M_E,z)-1);}
public constfunc f32  elu(f32 z,  f32 alpha) {return z > 0.0f ? z : alpha*(pow(M_E,z)-1);}
public constfunc f64  elu(f64 z,  f64 alpha) {return z > 0.0 ? z  : alpha*(pow(M_E,z)-1);}


public constfunc u8   relu(u8 z)  {return max(0,z);}
public constfunc s8   relu(s8 z)  {return max(0,z);}
public constfunc u16  relu(u16 z) {return max(0,z);}
public constfunc s16  relu(s16 z) {return max(0,z);}
public constfunc u32  relu(u32 z) {return max(0,z);}
public constfunc s32  relu(s32 z) {return max(0,z);}
public constfunc u64  relu(u64 z) {return max(0,z);}
public constfunc s64  relu(s64 z) {return max(0,z);}
public constfunc u128 relu(u128 z){return max(0,z);}
public constfunc s128 relu(s128 z){return max(0,z);}
public constfunc f16  relu(f16 z) {return max(0,z);}
public constfunc f32  relu(f32 z) {return max(0,z);}
public constfunc f64  relu(f64 z) {return max(0,z);}

public constfunc u8   sigmoid(u8 z)  {unreachable();}
public constfunc s8   sigmoid(s8 z)  {unreachable();}
public constfunc u16  sigmoid(u16 z) {unreachable();}
public constfunc s16  sigmoid(s16 z) {unreachable();}
public constfunc u32  sigmoid(u32 z) {unreachable();}
public constfunc s32  sigmoid(s32 z) {unreachable();}
public constfunc u64  sigmoid(u64 z) {unreachable();}
public constfunc s64  sigmoid(s64 z) {unreachable();}
public constfunc u128 sigmoid(u128 z){unreachable();}
public constfunc s128 sigmoid(s128 z){unreachable();}
public constfunc f16  sigmoid(f16 z) {return 1.0f / (1.0f + pow(M_E,-z));}
public constfunc f32  sigmoid(f32 z) {return 1.0f / (1.0f + pow(M_E,-z));}
public constfunc f64  sigmoid(f64 z) {return 1.0  / (1.0  + pow(M_E,-z));}