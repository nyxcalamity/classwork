#include "boundary.h"
#include "LBDefinitions.h"
#include "computeCellValues.h"

int inv(int i){
    return (Q-1)-i;
}

void treatBoundary(double *collideField, int* flagField, const double * const wallVelocity, int xlength){
    int x,nx,y,ny,z,nz,i,step=xlength+2;
    double density,dotProd;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
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