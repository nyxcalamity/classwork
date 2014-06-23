#include "collision.h"
#include "compute_cell_values.h"
#include "lbm_definitions.h"
#include "helper.h"


/** Computes the post-collision distribution functions according to the BGK update rule and
 *  stores the results again at the same position.
 */
void ComputePostCollisionDistributions(double *current_cell, double tau, const double *const feq){
    int i;
    for(i=0;i<Q_LBM;i++){
        current_cell[i]=current_cell[i]-(current_cell[i]-feq[i])/tau;
        
        /* Probability distribution function can not be less than 0 */
        if (current_cell[i] < 0)
            ERROR("Probability distribution function can not be negative.");
    }
}


void DoCollision(double *collide_field, int *flag_field, double tau, int xlength){
    double density, velocity[3], feq[Q_LBM], *currentCell;
    int x,y,z,step=xlength+2;
    
    for(x=1;x<step-1;x++){
        for(y=1;y<step-1;y++){
            for(z=1;z<step-1;z++){
                currentCell=&collide_field[Q_LBM*(x+y*step+z*step*step)];
                
                ComputeDensity(currentCell,&density);
                ComputeVelocity(currentCell,&density,velocity);
                ComputeFeq(&density,velocity,feq);
                ComputePostCollisionDistributions(currentCell,tau,feq);
                
                if(VERBOSE)
                    printf("(%d,%d,%d): density=%f, velocity=[%e,%e,%e]\n",x,y,z,
                            density,velocity[0],velocity[1],velocity[2]);
            }
        }
    }
}
