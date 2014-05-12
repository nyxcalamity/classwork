#include "collision.h"
#include "LBDefinitions.h"
#include "helper.h"

void computePostCollisionDistributions(double *currentCell, const double * const tau, 
        const double *const feq){
    int i;
    for(i=0;i<Q;i++){
        currentCell[i]=currentCell[i]-(currentCell[i]-feq[i])/(*tau);
        
        /* Probability distribution function can not be less than 0 */
        if (currentCell[i] < 0)
            ERROR("Probability distribution function can not be negative.");
    }
}

void doCollision(double *collideField, int *flagField, const double * const tau, int xlength){
    double density, velocity[3], feq[Q], *currentCell;
    int x,y,z,step=xlength+2;
    
    for(x=1;x<step-1;x++){
        for(y=1;y<step-1;y++){
            for(z=1;z<step-1;z++){
                currentCell=&collideField[Q*(x+y*step+z*step*step)];
                
                computeDensity(currentCell,&density);
                computeVelocity(currentCell,&density,velocity);
                computeFeq(&density,velocity,feq);
                computePostCollisionDistributions(currentCell,tau,feq);
                
                if(VERBOSE)
                    printf("(%d,%d,%d): density=%f, velocity=[%e,%e,%e]\n",x,y,z,
                            density,velocity[0],velocity[1],velocity[2]);
            }
        }
    }
}