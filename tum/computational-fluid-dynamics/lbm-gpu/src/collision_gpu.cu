#define BLOCK_SIZE 8

#include "collision_gpu.h"


__global__ void DoNothing(){
//	__syncthreads(); to use after reading data into shared memory
//    int i = blockDim.x*blockIdx.x + threadIdx.x;
//	Always test whether the indices are within bounds
//	if (i < n) do something;
//    cuPrintf("Hi there pretty! Wanna come up for a tea? My address is: block #%d thread #%d gridwise #%d",
//    		blockIdx.x, threadIdx.x, i);
}

void CudaTest(double *collide_field, size_t size){
	double *collide_field_d=NULL;

	cudaMalloc(&collide_field_d, size);
	cudaMemcpy(collide_field_d, collide_field, size, cudaMemcpyHostToDevice);
//	cudaMalloc(&d_a, n*sizeof(float));
//	cudaError_t e = cudaGetLastError();
//	if (e!=cudaSuccess){
//		cerr << "ERROR: " << cudaGetErrorString(e) << endl;
//		exit(1);
//	}

	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
//	dim3 grid = dim3( (w + block.x – 1) / block.x,
//	(h + block.y - 1) / block.y, 1 )
	DoNothing<<<1, dim_block>>>();
//	cudaDeviceSynchronize(); this is already in memcpy
	cudaMemcpy(collide_field, collide_field_d, size, cudaMemcpyDeviceToHost);
	cudaFree(collide_field_d);
}

void DoCollisionCuda(double *collide_field, int *flag_field, const double * const tau, int xlength){

}
