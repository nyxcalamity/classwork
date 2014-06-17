#include "collision_gpu.h"

__global__
void DoNothing(){
//    int i = blockDim.x*blockIdx.x + threadIdx.x;
//    cuPrintf("Hi there pretty! Wanna come up for a tea? My address is: block #%d thread #%d gridwise #%d",
//    		blockIdx.x, threadIdx.x, i);
}

void CudaTest(double *collide_field, size_t size){
	double *collide_field_d=NULL;

	//Ninja code
	cudaMalloc(&collide_field_d, size);
	cudaMemcpy(collide_field_d, collide_field, size, cudaMemcpyHostToDevice);
	DoNothing<<<1, 6>>>();
	cudaMemcpy(collide_field, collide_field_d, size, cudaMemcpyDeviceToHost);
	cudaFree(collide_field_d);
}
