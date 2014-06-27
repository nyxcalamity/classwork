#ifndef _COMPUTECELLVALUESGPU_H_
#define _COMPUTECELLVALUESGPU_H_

//TODO:move lattice constants to gpu constant memory
__device__ static const int LATTICE_VELOCITIES_D[19][3] = {
    {0,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{0,1,-1},{-1,-1,0},{0,-1,0},{1,-1,0},
    {-1,0,0}, {0,0,0},  {1,0,0}, {-1,1,0},{0,1,0}, {1,1,0},  {0,-1,1},{-1,0,1},
    {0,0,1},  {1,0,1},  {0,1,1}
};
__device__ static const float LATTICE_WEIGHTS_D[19] = {
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0,
    2.0/36.0, 12.0/36.0,2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    2.0/36.0, 1.0/36.0, 1.0/36.0
};

/**
 * Computes the density from the particle distribution functions stored at currentCell.
 * currentCell thus denotes the address of the first particle distribution function of the
 * respective cell. The result is stored in density.
 */
void ComputeDensityGpu(float *current_cell, float *density);

/**
 * Computes the velocity within currentCell and stores the result in velocity
 */
void ComputeVelocityGpu(float *current_cell, float *density, float *velocity);

/**
 * Computes the equilibrium distributions for all particle distribution functions of one
 * cell from density and velocity and stores the results in feq.
 */
void ComputeFeqGpu(float *density, float *velocity, float *feq);

#endif
