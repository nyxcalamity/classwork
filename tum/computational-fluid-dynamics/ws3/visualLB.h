#ifndef _VISUALLB_H_
#define _VISUALLB_H_

/** writes the density and velocity field (derived from the distributions in collideField)
 *  to a file determined by 'filename' and timestep 't'. You can re-use parts of the code
 *  from visual.c (VTK output for Navier-Stokes solver) and modify it for 3D datasets.
 */
void writeVtkOutput(const double * const collideField, const int * const flagField, const char * filename, unsigned int t, int *xlength);


/**
 * Function that prints out the point by point values of the provided field (4D).
 * @param field
 *          linerized 4D array, with (x,y,z,i)=Q_LBM*(x+y*(ncell+2)+z*(ncell+2)*(ncell+2))+i
 * @param ncell
 *          number of inner cells, the ones which are there before adding a boundary layer
 */
void printField(double *field, int *ncell);
void printFlagField(int *flagField, int *ncell);

#endif

