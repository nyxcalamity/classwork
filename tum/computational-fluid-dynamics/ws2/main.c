#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"
#include <tgmath.h>

int main(int argc, char *argv[]){
    double *collideField=NULL, *streamField=NULL, tau, velocityWall;
    int *flagField=NULL, xlength, timesteps, timestepsPerPlotting, num_cells;
    
    readParameters(&xlength,&tau,&velocityWall,&timesteps,&timestepsPerPlotting,argc,argv);
    
    num_cells = (xlength+2)*(xlength+2)*(xlength+2);
    collideField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    streamField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    flagField = malloc(num_cells*sizeof(*flagField));
    initialiseFields(collideField,streamField,flagField,xlength);
    
    /*
    for(int t = 0; t < timesteps; t++){
        double *swap=NULL;
        doStreaming(collideField,streamField,flagfield,xlength);
        swap = collideField;
        collideField = streamField;
        streamField = swap;
        doCollision(collideField,flagfield,&tau,xlength);
        treatBoundary(collideField,flagfield,velocityWall,xlength);
        if (t%timestepsPerPlotting==0){
            writeVtkOutput(collideField,flagfield,argv,t,xlength);
        }
    }
    */

    /* Free memory */
    free(collideField);
    free(streamField);
    free(flagField);
    
    return 0;
}

#endif