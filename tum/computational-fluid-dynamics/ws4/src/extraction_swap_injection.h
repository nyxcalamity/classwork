#ifndef EXTRACTION_SWAP_INJECTION_H_
#define EXTRACTION_SWAP_INJECTION_H_

#include "LBDefinitions.h"
#include <mpi.h>

/**
 * Function that does extraction step.
 * @param collideField
 *          linerized 4D array, with (x,y,z,i)=Q_LBM*(x+y*(x_sub_length+2)+z*(x_sub_length+2)*(x_sub_length+2))+i
 * @param flagField
 *          linearized 3D array with (x,y,z)=x+y*(x_sub_length+2)+z*(x_sub_length+2)*(x_sub_length+2)
 * @param sendBuffer
 *          linearized 3D array with (x,y,i)=Q_LBM*(x+y*(x_sub_length+2))+i
 * @param direction
 *          direction in which data should be sent
 * @param x_sub_length
 *          number of inner cells of subdomain, the ones which are there before adding a boundary layer
 */
void extractionMPI(double *collideField, double *sendBuffer, const int direction, const int x_sub_length);


/**
 * Function that does swap step.
 * @param sendBuffer
 *          linearized 3D array with (x,y,i)=Q_LBM*(x+y*(x_sub_length+2))+i
 * @param readBuffer
 *          linearized 3D array with (x,y,i)=Q_LBM*(x+y*(x_sub_length+2))+i
 * @param neighbor_cells
 *          array of the neighbor cells [0:left,1:rigth,2:top,3:bottom,4:front,5:back]
 * @param direction
 *          direction in which data should be sent
 * @param x_sub_length
 *          number of inner cells of subdomain, the ones which are there before adding a boundary layer
 */
void swapMPI(double *sendBuffer, double *readBuffer,const int * const neighbor_cells, const int direction, const int x_sub_length);


/**
 * Function that does injection step.
 * @param collideField
 *          linerized 4D array, with (x,y,z,i)=Q_LBM*(x+y*(x_sub_length+2)+z*(x_sub_length+2)*(x_sub_length+2))+i
 * @param flagField
 *          linearized 3D array with (x,y,z)=x+y*(x_sub_length+2)+z*(x_sub_length+2)*(x_sub_length+2)
 * @param readBuffer
 *          linearized 3D array with (x,y,i)=Q_LBM*(x+y*(x_sub_length+2))+i
 * @param direction
 *          direction in which data should be sent
 * @param x_sub_length
 *          number of inner cells of subdomain, the ones which are there before adding a boundary layer
 */
void injectionMPI(double *collideField, const int * const flagField, double *readBuffer, const int direction, const int x_sub_length);

#endif /* EXTRACTION_SWAP_INJECTION_H_ */
