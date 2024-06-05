#include <curand_kernel.h>

__device__ inline float randomFloat(curandState* randState) {
	return curand_uniform(randState);
}

__device__ inline float randomFloat(float a1, float a2, curandState* randState) {
	return randomFloat(randState) * (a2 - a1) + a1;
}