#ifndef _COLLISION_H_
#define _COLLISION_H_

#include "compute_cell_values.h"

/** computes the post-collision distribution functions according to the BGK update rule and
 *  stores the results again at the same position.
 */
void computePostCollisionDistributions(double *current_cell, const double * const tau, 
        const double * const feq);


/** carries out the whole local collision process. Computes density and velocity and
 *  equilibrium distributions. Carries out BGK update.
 */
void doCollision(double *collide_field, int *flag_field, const double * const tau, int xlength);
#endif

