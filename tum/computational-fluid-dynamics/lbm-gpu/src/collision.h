#ifndef _COLLISION_H_
#define _COLLISION_H_

/** carries out the whole local collision process. Computes density and velocity and
 *  equilibrium distributions. Carries out BGK update.
 */
void DoCollision(double *collide_field, int *flag_field, const double * const tau, int xlength);

#endif
