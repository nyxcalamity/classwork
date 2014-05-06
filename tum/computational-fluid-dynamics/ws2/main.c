#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"
#include <time.h>

/**
 * Function that prints out the point by point values of the provided field (4D).
 * @param field
 *          linerized 4D array, with (x,y,z,i)=Q*(x+y*(ncell+2)+z*(ncell+2)*(ncell+2))+i
 * @param ncell
 *          number of inner cells, the ones which are there before adding a boundary layer
 */
void printField(double *field, int ncell){
    int x,y,z,i,step=ncell+2;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                printf("(%d,%d,%d): ",x,y,z);
                for(i=0;i<Q;i++){
                    printf("%f ",field[Q*(x+y*step+z*step*step)+i]);
                }
                printf("\n");
            }
        }
    }
}

int main(int argc, char *argv[]){
    double *collideField=NULL, *streamField=NULL, *swap=NULL, tau, velocityWall[3], num_cells;
    int *flagField=NULL, xlength, t, timesteps, timestepsPerPlotting, mlups_exp=pow(10,6);
    clock_t mlups_time;
    
    readParameters(&xlength,&tau,velocityWall,&timesteps,&timestepsPerPlotting,argc,argv);
    
    num_cells = pow(xlength+2, D);
    collideField = malloc(Q*num_cells*sizeof(*collideField));
    streamField = malloc(Q*num_cells*sizeof(*collideField));
    flagField = malloc(num_cells*sizeof(*flagField));
    initialiseFields(collideField,streamField,flagField,xlength);
    
    for(t=0;t<timesteps;t++){
        mlups_time = clock();
        /* Copy pdfs from neighbouring cells into collide field */
        doStreaming(collideField,streamField,flagField,xlength);
        /* Perform the swapping of collide and stream fields */
        swap = collideField; collideField = streamField; streamField = swap;
        /* Compute post collision distributions */
        doCollision(collideField,flagField,&tau,xlength);
        /* Treat boundaries */
        treatBoundary(collideField,flagField,velocityWall,xlength);
        /* Print out the MLUPS value */
        mlups_time = clock()-mlups_time;
        printf("Time step: #%d MLUPS: %f\n", t, num_cells/(mlups_exp*(double)mlups_time/CLOCKS_PER_SEC));
        /* Print out vtk output if needed */
        if (t%timestepsPerPlotting==0)
            writeVtkOutput(collideField,flagField,"img/lbm-img",t,xlength);
        
        if(VERBOSE)
            printField(collideField, xlength);
    }

    /* Free memory */
    free(collideField);
    free(streamField);
    free(flagField);
    
    printf("Simulation complete.");
    return 0;
}

#endif