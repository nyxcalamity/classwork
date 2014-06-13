#ifndef _VISUALLB_H_
#define _VISUALLB_H_

/** writes the density and velocity field (derived from the distributions in collideField)
 *  to a file determined by 'filename' and timestep 't'. You can re-use parts of the code
 *  from visual.c (VTK output for Navier-Stokes solver) and modify it for 3D datasets.
 */
void writeVtkOutput(const double * const collideField, const int * const flagField, const char * filename, 
        unsigned int t, int xlength, int rank, int x_proc);

/**
 * Function that prints out the point by point values of the provided field (4D).
 * @param field
 *          linerized 4D array, with (x,y,z,i)=Q_LBM*(x+y*(ncell+2)+z*(ncell+2)*(ncell+2))+i
 * @param ncell
 *          number of inner cells, the ones which are there before adding a boundary layer
 */
void printField(double *field, int xlength);
void printFlagField(int *flagField, int xlength);


/**
 * Function that writes out the point by point values of the provided field (4D).
 * @param field
 *          linerized 4D array, with (x,y,z,i)=Q_LBM*(x+y*(ncell+2)+z*(ncell+2)*(ncell+2))+i
 * @param ncell
 *          number of inner cells, the ones which are there before adding a boundary layer
 */
void writeField(const double * const field, const char * filename, unsigned int t, const int xlength, const int rank);
void writeFlagField(const int * const flagField, const char * filename, const int xlength, const int rank);


/**
 * Function that writes out the point by point values of the provided field (4D).
 * @param buffer
 *          linerized 3D array, with (x,y,i)=Q_LBM*(x+y*(x_sub_length+2))+i
 * @param x_sub_length
 *          number of inner cells of subdomain, the ones which are there before adding a boundary layer
 */
void writeBuffer(const double * const buffer, const char * filename, const unsigned int t, const int x_sub_length, const int rank);

#endif

