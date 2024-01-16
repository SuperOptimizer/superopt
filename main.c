
#include "common.h"

#include "day/4/mcmc.h"

ARR(u32, 16, arr_u32)
VEC(u32, vec_u32)

int main(int argc, char** argv){
	printf("asdf\n");
	u8 _v[256];
	for(s32 i = 0; i <= 255; i++) _v[i]=i;

	u8vec bytevec = u8vec_new(256, _v);

	mcmc* mcmc = mcmc_new();
	mcmc_train(mcmc, bytevec);

	return 0;
}