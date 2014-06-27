#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

/** handles the boundaries in our simulation setup */
void TreatBoundary(float *collide_field, int* flag_field, const float * const wall_velocity,
        int xlength);
#endif
