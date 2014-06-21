#include "collision_gpu.h"
#include <stdio.h>

#define BLOCK_SIZE 16

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
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	DoNothing<<<1, dim_block>>>();
	cudaMemcpy(collide_field, collide_field_d, size, cudaMemcpyDeviceToHost);
	cudaFree(collide_field_d);
	printf("Completed CUDA part.\n");
}
