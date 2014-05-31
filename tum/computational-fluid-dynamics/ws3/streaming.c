#include "streaming.h"
#include "LBDefinitions.h"

void doStreaming(double *collide_field, double *stream_field, int *flag_field, int *xlength){
    int x, nx, y, ny, z, nz, i, step_x=xlength[0]+2, step_y=xlength[1]+2, step_z=xlength[2]+2;
    
    for(x=0;x<step_x;x++){
        for(y=0;y<step_y;y++){
            for(z=0;z<step_z;z++){
                if(flag_field[x+y*step_x+z*step_x*step_y]==FLUID){
                    for(i=0;i<Q_LBM;i++){
                        nx=x-LATTICEVELOCITIES[i][0];
                        ny=y-LATTICEVELOCITIES[i][1];
                        nz=z-LATTICEVELOCITIES[i][2];
                        
                        stream_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=
                                collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+i];
                    }
                }
            }
        }
    }
}

