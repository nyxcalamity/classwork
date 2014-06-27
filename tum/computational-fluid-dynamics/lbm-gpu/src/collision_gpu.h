#ifndef CUDA_CALLS_H_
#define CUDA_CALLS_H_

/** Carries out the whole local collision process. Computes density and velocity and
 *  equilibrium distributions. Carries out BGK update.
 */
void DoCollisionCuda(float *collide_field, int *flag_field, float tau, int xlength);

/**
 * Carries out the streaming step and writes the respective distribution functions from
 * collideField to streamField.
 */
void DoStreamingCuda(float *collide_field, float *stream_field, int *flag_field, int xlength);

#endif
