#include "initLB.h"
#include "helper.h"
#include <unistd.h>

int readParameters(int *xlength, double *tau, double *velocityWall, 
        int *timesteps, int *timestepsPerPlotting, int argc, char *argv[]){
    double *velocityWall1, *velocityWall2, *velocityWall3;
    
    if(argc<2)
        ERROR("Not enough arguments. At least a path to init file is required.");
    if(access(argv[1], R_OK) != 0)
        ERROR("Provided file path either doesn't exist or can not be read.");
    
    READ_DOUBLE(argv[1], *tau);
    
    velocityWall1=&velocityWall[0];
    velocityWall2=&velocityWall[1];
    velocityWall3=&velocityWall[2];
    READ_DOUBLE(argv[1], *velocityWall1);
    READ_DOUBLE(argv[1], *velocityWall2);
    READ_DOUBLE(argv[1], *velocityWall3);
    
    READ_INT(argv[1], *xlength);
    READ_INT(argv[1], *timesteps);
    READ_INT(argv[1], *timestepsPerPlotting);
    
    return 1;
}


void initialiseFields(double *collideField, double *streamField, int *flagField, int xlength){
    int x,y,z,i,step=xlength+2;
    
    /* NOTE: We use z=xlength+1 as the moving wall */
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                /* Initializing flags */
                if(z == xlength+1)
                    flagField[x+y*step+z*step*step]=MOVING_WALL;
                else if(x == 0 || x == xlength+1 || y == 0 || y == xlength+1 || z == 0)
                    flagField[x+y*step+z*step*step]=NO_SLIP;
                else
                    flagField[x+y*step+z*step*step]=FLUID;
                
                /* Initializing distributions for stream and collide fields */
                for(i=0;i<Q;i++){
                    /* NOTE:Stream field is initilized to 0s because that helps to 
                     * track down mistakes and has no impact whatsoever to on the 
                     * computation further on.
                     */
                    streamField[Q*(x+y*step+z*step*step)+i]=0;
                    collideField[Q*(x+y*step+z*step*step)+i]=LATTICEWEIGHTS[i];
                }
            }
        }
    }
}