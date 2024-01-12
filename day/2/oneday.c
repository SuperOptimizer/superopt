#include "oneday.h"

int asdf(r8 reg) {
	return reg.u8[0];
}


int main(int argc, char** argv) {
	f32x32 asdf;
	r8 reg;
	reg.u8[0] = 1;
	printf("%lu\n",sizeof asdf);
	return 0;
}