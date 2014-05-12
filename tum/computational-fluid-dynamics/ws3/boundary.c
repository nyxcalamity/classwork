#include "boundary.h"
#include "LBDefinitions.h"
#include "computeCellValues.h"

/**
 * Inverts the value of the lattice index in order to find the vector opposite to the provided one.
 * @param i
 *      index to inverse
 * @return 
 *      inversed index
 */
int inv(int i){
    return (Q-1)-i;
}

void treatBoundary(double *collideField, int* flagField, const double * const wallVelocity, int xlength){
    int x,nx,y,ny,z,nz,i,step=xlength+2;
    double density,dotProd;
    /* NOTE:you can improve the performance of this function by looping over only the boundaries,
     * it will save a number memory access calls, comparisons and control jumps.
     * 
     * However, for our case the performance gain was so insignificant that we decided to
     * stick to this implementation which is more clear and has all the formulas in one place
     * which reduces the probability of error. 
     * 
     * If you are interested in a more efficient implementation, please check 
     * https://github.com/POWER-Morzh/CFDLab02/blob/master/boundary.c
     * where we implemented this function in two ways (second one in comments) and
     * we will gladly substitute current implementation with the mentioned one.
     *  */
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                if(flagField[x+y*step+z*step*step]!=FLUID){
                    for(i=0;i<Q;i++){
                        nx=x+LATTICEVELOCITIES[i][0];
                        ny=y+LATTICEVELOCITIES[i][1];
                        nz=z+LATTICEVELOCITIES[i][2];

                        /* We don't need the values outside of our extended domain */
                        if(0<nx && nx<step-1 && 0<ny && ny<step-1 && 0<nz && nz<step-1){
                            if (flagField[x+y*step+z*step*step]==MOVING_WALL){
                                /* Compute density in the neighbour cell */
                                computeDensity(&collideField[Q*(nx+ny*step+nz*step*step)],&density);
                                /* Compute dot product */
                                dotProd=LATTICEVELOCITIES[i][0]*wallVelocity[0]+
                                        LATTICEVELOCITIES[i][1]*wallVelocity[1]+
                                        LATTICEVELOCITIES[i][2]*wallVelocity[2];
                                /* Assign the boudary cell value */
                                collideField[Q*(x+y*step+z*step*step)+i]=
                                        collideField[Q*(nx+ny*step+nz*step*step)+inv(i)]+
                                        2*LATTICEWEIGHTS[i]*density*C_S_POW2_INV*dotProd;
                            }else if(flagField[x+y*step+z*step*step]==NO_SLIP){
                                collideField[Q*(x+y*step+z*step*step)+i]=
                                        collideField[Q*(nx+ny*step+nz*step*step)+inv(i)];
                            }
                        }
                    }
                }
            }
        }
    }
}