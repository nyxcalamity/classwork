#include <math.h>
#include <stdio.h>
#include "lbm_definitions.h"
#include "collision_gpu.h"

//restricts 3D blocks to have 512 threads (limits: 512 CC<2.x; 1024 CC>2.x)
#define BLOCK_SIZE 8

/**
 * Checks the returned cudaError_t and prints corresponding message in case of error.
 */
#define cudaErrorCheck(ans){ cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__constant__ double tau_d;
__constant__ int xlength_d;
//TODO:move lattice constants to gpu constant memory
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


/**
 * Computes the density from the particle distribution functions stored at currentCell.
 * currentCell thus denotes the address of the first particle distribution function of the
 * respective cell. The result is stored in density.
 */
__device__ void ComputeDensity(double *current_cell, double *density){
    int i; *density=0;
    for(i=0;i<Q_LBM;i++)
        *density+=current_cell[i];

    /* Density should be close to a unit (ρ~1) */
//    if((*density-1.0)>EPS)
//        ERROR("Density dropped below error tolerance.");
}


/**
 * Computes the velocity within currentCell and stores the result in velocity
 */
__device__ void ComputeVelocity(double *current_cell, double *density, double *velocity){
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


/**
 * Computes the equilibrium distributions for all particle distribution functions of one
 * cell from density and velocity and stores the results in feq.
 */
__device__ void ComputeFeq(double *density, double *velocity, double *feq){
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


/**
 * Computes the post-collision distribution functions according to the BGK update rule and
 * stores the results again at the same position.
 */
__device__ void ComputePostCollisionDistributions(double *current_cell, double *feq){
    int i;
    for(i=0;i<Q_LBM;i++){
        current_cell[i]=current_cell[i]-(current_cell[i]-feq[i])/tau_d;

        /* Probability distribution function can not be less than 0 */
//        if (current_cell[i] < 0)
//            ERROR("Probability distribution function can not be negative.");
    }
}

/**
 * Performs the actual collision computation
 */
__global__ void DoColision(double *collide_field_d){
	//	__syncthreads(); to use after reading data into shared memory
	double density, velocity[D_LBM], feq[Q_LBM], *currentCell;
	int x = 1+threadIdx.x+blockIdx.x*blockDim.x;
	int y = 1+threadIdx.y+blockIdx.y*blockDim.y;
	int z = 1+threadIdx.z+blockIdx.z*blockDim.z;
	int step = xlength_d+2;
	int idx = x+y*step+z*step*step;

	//check that indices are within the bounds since there could be more threads than needed
	if (x<(step-1) && y<(step-1) && z<(step-1)){
		currentCell=&collide_field_d[Q_LBM*idx];
		ComputeDensity(currentCell,&density);
		ComputeVelocity(currentCell,&density,velocity);
		ComputeFeq(&density,velocity,feq);
		ComputePostCollisionDistributions(currentCell,feq);
	}
}


void DoCollisionCuda(double *collide_field, int *flag_field, double tau, int xlength){
	double *collide_field_d=NULL;
	int num_cells = pow(xlength+2, D_LBM);
	size_t collide_field_size = Q_LBM*num_cells*sizeof(double);

	//initialize working data
	cudaErrorCheck(cudaMalloc(&collide_field_d, collide_field_size));
	cudaErrorCheck(cudaMemcpy(collide_field_d, collide_field, collide_field_size, cudaMemcpyHostToDevice));

	//initialize constant data
	cudaErrorCheck(cudaMemcpyToSymbol(tau_d, &tau, sizeof(double), 0, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyToSymbol(xlength_d, &xlength, sizeof(int), 0, cudaMemcpyHostToDevice));

	//define grid structure
	//NOTE:redundant threads for boundary cells are not accounted for
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((xlength+block.x-1)/block.x, (xlength+block.y-1)/block.y, (xlength+block.z-1)/block.z);

	//perform collision
	DoColision<<<grid,block>>>(collide_field_d);
	cudaErrorCheck(cudaPeekAtLastError());

	//copy data back to host
	cudaErrorCheck(cudaMemcpy(collide_field, collide_field_d, collide_field_size, cudaMemcpyDeviceToHost));

	//free device memory
	cudaErrorCheck(cudaFree(collide_field_d));
}


/**
 * Performs the actual streaming computation
 */
__global__ void DoStreaming(double *stream_field_d, double *collide_field_d){
	//	__syncthreads(); to use after reading data into shared memory
	int x = 1+threadIdx.x+blockIdx.x*blockDim.x;
	int y = 1+threadIdx.y+blockIdx.y*blockDim.y;
	int z = 1+threadIdx.z+blockIdx.z*blockDim.z;
	int step = xlength_d+2, idx = x+y*step+z*step*step, nx, ny, nz, i;

	//check that indices are within the bounds since there could be more threads than needed
	if (x<(step-1) && y<(step-1) && z<(step-1)){
		for(i=0;i<Q_LBM;i++){
			nx=x-LATTICE_VELOCITIES_D[i][0];
			ny=y-LATTICE_VELOCITIES_D[i][1];
			nz=z-LATTICE_VELOCITIES_D[i][2];

			stream_field_d[Q_LBM*idx+i]=
					collide_field_d[Q_LBM*(nx+ny*step+nz*step*step)+i];
		}
	}
}


void DoStreamingCuda(double *collide_field, double *stream_field, int *flag_field, int xlength){
	double *collide_field_d=NULL, *stream_field_d=NULL;
	int num_cells = pow(xlength+2, D_LBM);
	size_t computational_field_size = Q_LBM*num_cells*sizeof(double);

	//initialize working data
	cudaErrorCheck(cudaMalloc(&collide_field_d, computational_field_size));
	cudaErrorCheck(cudaMemcpy(collide_field_d, collide_field, computational_field_size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMalloc(&stream_field_d, computational_field_size));
	cudaErrorCheck(cudaMemcpy(stream_field_d, stream_field, computational_field_size, cudaMemcpyHostToDevice));

	//initialize constant data
	cudaErrorCheck(cudaMemcpyToSymbol(xlength_d, &xlength, sizeof(int), 0, cudaMemcpyHostToDevice));

	//define grid structure
	//NOTE:redundant threads for boundary cells are not accounted for
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((xlength+block.x-1)/block.x, (xlength+block.y-1)/block.y, (xlength+block.z-1)/block.z);

	//perform streaming
	DoStreaming<<<grid,block>>>(stream_field_d, collide_field_d);
	cudaErrorCheck(cudaPeekAtLastError());

	//copy data back to host
	cudaErrorCheck(cudaMemcpy(stream_field, stream_field_d, computational_field_size, cudaMemcpyDeviceToHost));

	//free device memory
	cudaErrorCheck(cudaFree(collide_field_d));
	cudaErrorCheck(cudaFree(stream_field_d));
}
