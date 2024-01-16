#pragma once

#include "common.h"

//byte level markov chain
// s32 mcmc [256][256][256] 1st char, 2nd char, # occurrences of 3rd char

typedef struct mcmc {
	s32 chain[256][256][256];
} mcmc aligned(16);

public mcmc* mcmc_new(void) {
	mcmc* ret = malloc(sizeof(mcmc));
	for(int x = 0; x < 256; x++) {
		for(int y = 0; y < 256; y++) {
			for(int z = 0; z < 256; z++) {
				ret->chain[x][y][z] = 0.0f;
			}
		}
	}
	return ret;
}

public void mcmc_train(mcmc* mcmc, u8vec data) {
	for(int i = 0; i < len(data) - 2; i++) {
		auto x = at(data,i);
		auto y = at(data,i+1);
		auto z = at(data,i+2);
		mcmc->chain[x][y][z]++;
	}
}

public u8 mcmc_predict(mcmc* mcmc, u8 x, u8 y) {
	s32 occurrences = 0;
	for(int i = 0; i < 256; i++) {
		occurrences+= mcmc->chain[x][y][i];
	}
	s32 tgt = randint() % occurrences;
	s32 lower = 0;
	for(int i = 0; i < 256; i++) {
		auto higher =  mcmc->chain[x][y][i] + lower;
		if(tgt >= lower && tgt < higher) {
			return i;
		}
		lower = higher;
	}
	unreachable();
}