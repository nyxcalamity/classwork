#include "initLB.h"
#include "helper.h"
#include <unistd.h>

int readParameters(int *xlength, double *tau, double *velocityWall, 
        int *timesteps, int *timestepsPerPlotting, int argc, char *argv[]){
    double velocityWall1, velocityWall2, velocityWall3;
    
    if(argc<2)
        ERROR("Not enough arguments. At least a path to init file is required.");
    if(access(argv[1], R_OK) != 0)
        ERROR("Provided file path either doesn't exist or can not be read.");
    
    READ_DOUBLE(argv[1], *tau);
    READ_DOUBLE(argv[1], velocityWall1);
    READ_DOUBLE(argv[1], velocityWall2);
    READ_DOUBLE(argv[1], velocityWall3);
    
    velocityWall[1]=velocityWall1;
    velocityWall[2]=velocityWall2;
    velocityWall[3]=velocityWall3;
    
    READ_INT(argv[1], *xlength);
    READ_INT(argv[1], *timesteps);
    READ_INT(argv[1], *timestepsPerPlotting);
    
    return 1;
}


void initialiseFields(double *collideField, double *streamField, int *flagField, int xlength){
    int x,y,z,i,step=xlength+2;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                /* Initializing flags */
                if(y == xlength+1)
                    flagField[x+y*step+z*step*step]=MOVING_WALL;
                else if(y == 0 || x == 0 || x == xlength+1 || z == 0 || z == xlength+1)
                    flagField[x+y*step+z*step*step]=NO_SLIP;
                else
                    flagField[x+y*step+z*step*step]=FLUID;
                
                /* Initializing distributions for stream and collide fields */
                for(i=0;i<Q;i++){
                    if(i==9){
                        streamField[Q*(x+y*step+z*step*step)+i]=12/36;
                        collideField[Q*(x+y*step+z*step*step)+i]=12/36;
                    }else if(i==2 || i==6 || i==8 || i==10 || i==12 || i==16){
                        streamField[Q*(x+y*step+z*step*step)+i]=2/36;
                        collideField[Q*(x+y*step+z*step*step)+i]=2/36;
                    }else{
                        streamField[Q*(x+y*step+z*step*step)+i]=1/36;
                        collideField[Q*(x+y*step+z*step*step)+i]=1/36;
                    }
                }
            }
        }
    }
}