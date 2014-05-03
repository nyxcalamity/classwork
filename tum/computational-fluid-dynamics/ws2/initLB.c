#include "initLB.h"
#include "helper.h"
#include <unistd.h>

int readParameters(int *xlength, double *tau, double *velocityWall, 
        int *timesteps, int *timestepsPerPlotting, int argc, char *argv[]){
    if(argc<2)
        ERROR("Not enough arguments. At least a path to init file is required.");
    if(access(argv[1], R_OK) != 0)
        ERROR("Provided file path either doesn't exist or can not be read.");
    
    READ_DOUBLE(argv[1], *tau);
    READ_DOUBLE(argv[1], *velocityWall);
    
    READ_INT(argv[1], *xlength);
    READ_INT(argv[1], *timesteps);
    READ_INT(argv[1], *timestepsPerPlotting);
    
    return 1;
}


void initialiseFields(double *collideField, double *streamField, int *flagField, int xlength){
    int x,y,z,i;
    
    for(x=0;x<xlength+2;x++){
        for(y=0;y<xlength+2;y++){
            for(z=0;z<xlength+2;z++){
                /* Initializing flags */
                if(y == xlength+1)
                    flagField[x+y*xlength+z*xlength*xlength]=MOVING_WALL;
                else if(y == 0 || x == 0 || x == xlength+1 || z == 0 || z == xlength+1)
                    flagField[x+y*xlength+z*xlength*xlength]=NO_SLIP;
                else
                    flagField[x+y*xlength+z*xlength*xlength]=FLUID;
                
                /* Initializing distributions for stream field */
                for(i=0;i<Q_LBM;i++){
                    if(i==9)
                        streamField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=12/36;
                    else if(i==2 || i==6 || i==8 || i==10 || i==12 || i==16)
                        streamField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=2/36;
                    else
                        streamField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=1/36;
                }
                
                /* Initializing distributions for collide field */
                for(i=0;i<Q_LBM;i++){
                    if(i==9)
                        collideField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=12/36;
                    else if(i==2 || i==6 || i==8 || i==10 || i==12 || i==16)
                        collideField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=2/36;
                    else
                        collideField[Q_LBM*(x+y*xlength+z*xlength*xlength)+i]=1/36;
                }
            }
        }
    }
}