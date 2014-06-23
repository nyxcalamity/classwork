#ifndef CUDA_CALLS_H_
#define CUDA_CALLS_H_

/** Carries out the whole local collision process. Computes density and velocity and
 *  equilibrium distributions. Carries out BGK update.
 */
void DoCollisionCuda(double *collide_field, int *flag_field, double tau, int xlength);

#endif
