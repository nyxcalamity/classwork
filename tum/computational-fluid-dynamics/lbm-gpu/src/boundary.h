#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

/** handles the boundaries in our simulation setup */
void TreatBoundary(double *collide_field, int* flag_field, const double * const wall_velocity,
        int xlength);
#endif
