#ifndef CUDA_CALLS_H_
#define CUDA_CALLS_H_

/** CUDA test function that paerforms a single empty call */
void CudaTest(double *collide_field, size_t size);

/** Carries out the whole local collision process. Computes density and velocity and
 *  equilibrium distributions. Carries out BGK update.
 */
void DoCollisionCuda(double *collide_field, int *flag_field, const double * const tau, int xlength);

#endif
