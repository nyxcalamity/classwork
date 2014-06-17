#include "gpu_utils.h"


int hasCudaGpu(){
	int devices = 0;
	cudaError_t err = cudaGetDeviceCount(&devices);
	return (devices > 0 && err == cudaSuccess) ? 1 : 0;
}
