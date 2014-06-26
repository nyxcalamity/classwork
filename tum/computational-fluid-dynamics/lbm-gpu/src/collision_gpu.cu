#include <math.h>
#include <stdio.h>
#include "lbm_definitions.h"
#include "collision_gpu.h"

//restricts 3D blocks to have 512 threads (max 1024)
#define BLOCK_SIZE 512

#define cudaErrorCheck(ans){ cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__constant__ double tau_d;
__constant__ int xlength_d, num_cells_d;
//TODO:move lattice constants to gpu memory
__device__ static const int LATTICE_VELOCITIES_D[19][3] = {
    {0,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{0,1,-1},{-1,-1,0},{0,-1,0},{1,-1,0},
    {-1,0,0}, {0,0,0},  {1,0,0}, {-1,1,0},{0,1,0}, {1,1,0},  {0,-1,1},{-1,0,1},
    {0,0,1},  {1,0,1},  {0,1,1}
};
__device__ static const double LATTICE_WEIGHTS_D[19] = {
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0,
    2.0/36.0, 12.0/36.0,2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    2.0/36.0, 1.0/36.0, 1.0/36.0
};


/** computes the density from the particle distribution functions stored at currentCell.
 *  currentCell thus denotes the address of the first particle distribution function of the
 *  respective cell. The result is stored in density.
 */
__device__ void ComputeDensityCuda(double *current_cell, double *density){
    int i; *density=0;
    for(i=0;i<Q_LBM;i++)
        *density+=current_cell[i];

    /* Density should be close to a unit (ρ~1) */
//    if((*density-1.0)>EPS)
//        ERROR("Density dropped below error tolerance.");
}


/** computes the velocity within currentCell and stores the result in velocity */
__device__ void ComputeVelocityCuda(const double * const current_cell, const double * const density,
		double *velocity){
    int i;
    velocity[0]=0;
    velocity[1]=0;
    velocity[2]=0;

    for(i=0;i<Q_LBM;i++){
        velocity[0]+=current_cell[i]*LATTICE_VELOCITIES_D[i][0];
        velocity[1]+=current_cell[i]*LATTICE_VELOCITIES_D[i][1];
        velocity[2]+=current_cell[i]*LATTICE_VELOCITIES_D[i][2];
    }

    velocity[0]/=*density;
    velocity[1]/=*density;
    velocity[2]/=*density;
}


/** computes the equilibrium distributions for all particle distribution functions of one
 *  cell from density and velocity and stores the results in feq.
 */
__device__ void ComputeFeqCuda(const double * const density, const double * const velocity, double *feq){
    int i;
    double s1, s2, s3;
    for(i=0;i<Q_LBM;i++){
        s1 = LATTICE_VELOCITIES_D[i][0]*velocity[0]+LATTICE_VELOCITIES_D[i][1]*velocity[1]+
        		LATTICE_VELOCITIES_D[i][2]*velocity[2];
        s2 = s1*s1;
        s3 = velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2];

        feq[i]=LATTICE_WEIGHTS_D[i]*(*density)*(1+s1*C_S_POW2_INV+s2*C_S_POW4_INV/2.0-s3*C_S_POW2_INV/2.0);

        /* Probability distribution function can not be less than 0 */
//        if (feq[i] < 0)
//            ERROR("Probability distribution function can not be negative.");
    }
}


/** Computes the post-collision distribution functions according to the BGK update rule and
 *  stores the results again at the same position.
 */
__device__ void ComputePostCollisionDistributionsCuda(double *current_cell, const double * const feq){
    int i;
    for(i=0;i<Q_LBM;i++){
        current_cell[i]=current_cell[i]-(current_cell[i]-feq[i])/tau_d;

        /* Probability distribution function can not be less than 0 */
//        if (current_cell[i] < 0)
//            ERROR("Probability distribution function can not be negative.");
    }
}


__global__ void DoColision(double *collide_field_d, int *flag_field_d){
	//	__syncthreads(); to use after reading data into shared memory
	double density, velocity[3], feq[Q_LBM], *currentCell;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

//	int x = threadIdx.x+blockIdx.x*blockDim.x;
//	int y = threadIdx.y+blockIdx.y*blockDim.y;
//	int z = threadIdx.z+blockIdx.z*blockDim.z;
//	double density, velocity[D_LBM], feq[Q_LBM], *currentCell;
//	int step=xlength_d+2;
//	int idx = x+y*step+z*step*step;

//	if (0<x && x<(step-1) && 0<y && y<(step-1) && 0<z && z<(step-1) && !error_d[0] && !error_d[1] && !error_d[2]){
	//check that indices are within the bounds since there could be more threads than needed
	if(flag_field_d[idx] == FLUID && idx < num_cells_d){
		currentCell=&collide_field_d[Q_LBM*idx];
		ComputeDensityCuda(currentCell,&density);
		ComputeVelocityCuda(currentCell,&density,velocity);
		ComputeFeqCuda(&density,velocity,feq);
		ComputePostCollisionDistributionsCuda(currentCell,feq);
	}
}


void DoCollisionCuda(double *collide_field, int *flag_field, double tau, int xlength){
	double *collide_field_d=NULL;
	int *flag_field_d=NULL, num_cells = pow(xlength+2, D_LBM);
	size_t collide_field_size = Q_LBM*num_cells*sizeof(double);
	size_t flag_field_size = num_cells*sizeof(int);

	//initialize working data
	cudaErrorCheck(cudaMalloc(&collide_field_d, collide_field_size));
	cudaErrorCheck(cudaMemcpy(collide_field_d, collide_field, collide_field_size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMalloc(&flag_field_d, flag_field_size));
	cudaErrorCheck(cudaMemcpy(flag_field_d, flag_field, flag_field_size, cudaMemcpyHostToDevice));

	//initialize constant data
	cudaErrorCheck(cudaMemcpyToSymbol(tau_d, &tau, sizeof(double), 0, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyToSymbol(xlength_d, &xlength, sizeof(int), 0, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyToSymbol(num_cells_d, &num_cells, sizeof(int), 0, cudaMemcpyHostToDevice));

	//define grid structure
//	dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
//	dim3 grid((xlength+2+block.x-1)/block.x, (xlength+2+block.y-1)/block.y, (xlength+2+block.z-1)/block.z);

	//perform collision
	DoColision<<< ((num_cells+BLOCK_SIZE-1)/BLOCK_SIZE), BLOCK_SIZE >>>(collide_field_d, flag_field_d);

	cudaErrorCheck(cudaPeekAtLastError());
	//	cudaDeviceSynchronize(); this is already in memcpy
	//copy data back to host
	cudaErrorCheck(cudaMemcpy(collide_field, collide_field_d, collide_field_size, cudaMemcpyDeviceToHost));

	//free device memory
	cudaErrorCheck(cudaFree(collide_field_d));
	cudaErrorCheck(cudaFree(flag_field_d));
}
