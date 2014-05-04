#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"

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
    int *flagField=NULL, xlength, t, timesteps, timestepsPerPlotting;
    char *vtkOutputFile = "data/lbm-img";
    
    readParameters(&xlength,&tau,velocityWall,&timesteps,&timestepsPerPlotting,argc,argv);
    
    num_cells = pow(xlength+2, D);
    collideField = malloc(Q*num_cells*sizeof(*collideField));
    streamField = malloc(Q*num_cells*sizeof(*collideField));
    flagField = malloc(num_cells*sizeof(*flagField));
    initialiseFields(collideField,streamField,flagField,xlength);
    
    printf("Collide field after initialization....\n");
    /*printField(collideField, xlength);*/
    
    for(t=0;t<timesteps;t++){
        printf("Performing iteration #%d\n",t);
        /* Copy pdfs from neighbouring cells into collide field */
        doStreaming(collideField,streamField,flagField,xlength);
        /* Perform the swapping of collide and stream fields */
        swap = collideField; collideField = streamField; streamField = swap;
        /* Perform collision */
        doCollision(collideField,flagField,&tau,xlength);
        /* Treat boundaries */
        treatBoundary(collideField,flagField,velocityWall,xlength);
        /* Print out vtk output if needed */
        if (t%timestepsPerPlotting==0)
            writeVtkOutput(collideField,flagField,vtkOutputFile,t,xlength);
    }

    /* Free memory */
    free(collideField);
    free(streamField);
    free(flagField);
    
    printf("Simulation complete.");
    return 0;
}

#endif