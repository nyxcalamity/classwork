#include <math.h>
#include "lbm_definitions.h"
#include "collision_gpu.h"

//restricts 3D blocks to have 512 threads (max 1024)
#define BLOCK_SIZE 8

__constant__ double tau_d, xlength_d;
//TODO:Move lattice constants to constant memory
//__constant__ int LATTICE_VELOCITIES_D[Q_LBM];
//LATTICE_WEIGHTS_D[D_LBM][Q_LBM]

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
__device__ void ComputeDensity(double *current_cell, double *density){
    int i; *density=0;
    for(i=0;i<Q_LBM;i++)
        *density+=current_cell[i];

    /* Density should be close to a unit (ρ~1) */
//    if((*density-1.0)>EPS)
//        ERROR("Density dropped below error tolerance.");
}


/** computes the velocity within currentCell and stores the result in velocity */
__device__ void ComputeVelocity(double *current_cell, double *density, double *velocity){
    int i;
    /* NOTE:Indeces are hardcoded because of the possible performance gains and since
     * we do not have alternating D */
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
__device__ void ComputeFeq(double *density, double * velocity, double *feq){
    int i;
    double s1, s2, s3; /* summands */
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
__device__ void ComputePostCollisionDistributions(double *current_cell, double *feq){
    int i;
    for(i=0;i<Q_LBM;i++){
        current_cell[i]=current_cell[i]-(current_cell[i]-feq[i])/tau_d;

        /* Probability distribution function can not be less than 0 */
//        if (current_cell[i] < 0)
//            ERROR("Probability distribution function can not be negative.");
    }
}


__global__ void DoColision(double *collide_field_d){
	//	__syncthreads(); to use after reading data into shared memory
	//	Always test whether the indices are within bounds
	//	if (i < n) do something;
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	int z = threadIdx.z+blockIdx.z*blockDim.z;
	double density, velocity[3], feq[Q_LBM], *currentCell;
	int step=xlength_d+2;

	currentCell=&collide_field_d[Q_LBM*(x+y*step+z*step*step)];
	ComputeDensity(currentCell,&density);
	ComputeVelocity(currentCell,&density,velocity);
	ComputeFeq(&density,velocity,feq);
	ComputePostCollisionDistributions(currentCell,feq);
}


void DoCollisionCuda(double *collide_field, int *flag_field, double tau, int xlength){
	double *collide_field_d=NULL;
	int num_cells = pow(xlength+2, D_LBM);
	size_t size = Q_LBM*num_cells*sizeof(double);

	//initialize working data
	cudaMalloc(&collide_field_d, size);
	//	cudaMalloc(&d_a, n*sizeof(float));
	//	cudaError_t e = cudaGetLastError();
	//	if (e!=cudaSuccess){
	//		cerr << "ERROR: " << cudaGetErrorString(e) << endl;
	//		exit(1);
	//	}
	cudaMemcpy(collide_field_d, collide_field, size, cudaMemcpyHostToDevice);
	//initialize constant data
	cudaMemcpyToSymbol(tau_d, &tau, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(xlength_d, &xlength, sizeof(int), 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(LATTICE_VELOCITIES_D, &LATTICE_VELOCITIES, D_LBM*Q_LBM*sizeof(int), 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(LATTICE_WEIGHTS_D, &LATTICE_WEIGHTS, Q_LBM*sizeof(double), 0, cudaMemcpyHostToDevice);

	//define grid structure
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((xlength+block.x-1)/block.x, (xlength+block.y-1)/block.y, (xlength+block.z-1)/block.z);

	//perform collision
	DoColision<<<grid, block>>>(collide_field_d);

	//	cudaDeviceSynchronize(); this is already in memcpy
	//copy data back to host
	cudaMemcpy(collide_field, collide_field_d, size, cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(collide_field_d);
}
