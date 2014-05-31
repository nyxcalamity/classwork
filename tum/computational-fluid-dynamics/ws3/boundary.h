#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

/** handles the boundaries in our simulation setup */
void treatBoundary(double *collideField, int* flagField, const double * const wallVelocity,
		           int *xlength, const double densityRef, const double densityIn);

int isVelociryPerpendicular(const int i);


/*
 * The function which fills up two arrays to use them in the FREE_SLIP boundary.
 * @param respectIndex
 *         is an index of the pdf which velocity direction is directed on the neighbor cell
 * @param normalVelocities
 *         is the array of the size 5, which will be filled up with indexes of pdf's, which velocities can be mirrored
 *         in the neighbor cell. The neighbor cell is the cell in which velocity of respectIndex is directed from boundary cell.
 * @param mirroredVelocities
 *         is the array of the size 5, which will be filled up with indexes of pdf's, which are mirrored from the normalCells array.
 *         The order of pdf indexes in the mirroredCells responses to mirrored indexes of the normalCells.
 */
void findMirroredVelocities(const int respectIndex, int *normalVelocities, int *mirroredVelocities);

#endif

