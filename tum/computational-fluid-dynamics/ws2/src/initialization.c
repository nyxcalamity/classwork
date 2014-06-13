#include "initialization.h"
#include "helper.h"
#include <unistd.h>

int readParameters(int *xlength, double *tau, double *velocity_wall, int *timesteps, 
        int *timesteps_per_plotting, int argc, char *argv[]){
    double *velocity_wall_1, *velocity_wall_2, *velocity_wall_3;
    
    if(argc<2)
        ERROR("Not enough arguments. At least a path to init file is required.");
    if(access(argv[1], R_OK) != 0)
        ERROR("Provided file path either doesn't exist or can not be read.");
    
    READ_DOUBLE(argv[1], *tau);
    
    velocity_wall_1=&velocity_wall[0];
    velocity_wall_2=&velocity_wall[1];
    velocity_wall_3=&velocity_wall[2];
    READ_DOUBLE(argv[1], *velocity_wall_1);
    READ_DOUBLE(argv[1], *velocity_wall_2);
    READ_DOUBLE(argv[1], *velocity_wall_3);
    
    READ_INT(argv[1], *xlength);
    READ_INT(argv[1], *timesteps);
    READ_INT(argv[1], *timesteps_per_plotting);
    
    return 1;
}


void initialiseFields(double *collide_field, double *stream_field, int *flag_field, int xlength){
    int x,y,z,i,step=xlength+2;
    
    /* NOTE: We use z=xlength+1 as the moving wall */
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                /* Initializing flags */
                if(z == xlength+1)
                    flag_field[x+y*step+z*step*step]=MOVING_WALL;
                else if(x == 0 || x == xlength+1 || y == 0 || y == xlength+1 || z == 0)
                    flag_field[x+y*step+z*step*step]=NO_SLIP;
                else
                    flag_field[x+y*step+z*step*step]=FLUID;
                
                /* Initializing distributions for stream and collide fields */
                for(i=0;i<Q_LBM;i++){
                    /* NOTE:Stream field is initilized to 0s because that helps to 
                     * track down mistakes and has no impact whatsoever to on the 
                     * computation further on.
                     */
                    stream_field[Q_LBM*(x+y*step+z*step*step)+i]=0;
                    collide_field[Q_LBM*(x+y*step+z*step*step)+i]=LATTICEWEIGHTS[i];
                }
            }
        }
    }
}