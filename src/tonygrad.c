#include "common.h"
#include "tonygrad.h"

int main(int argc, char** argv) {
  s32 dims [] = {1024,0,0,0,0,0,0,0};
  auto a = tensor_new(DT_F32,dims_new(1,dims));
  auto b = tensor_new(DT_F32,dims_new(1,dims));

  for(int i = 0; i < 1024; i++) {
    set(a,i,1.0f);
    set(b,i,1.0f);
    get_f32(a,i);
    printf("%f\n",get_f32(a,i));
  }
  auto ret = add(a,b, DT_F32);
  for(int i = 0; i < 1024; i++) {
    printf("%f ",get_f32(ret,i));
    if(i % 16 == 15)printf("\n");
  }


}