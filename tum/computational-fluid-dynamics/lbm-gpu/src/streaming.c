#include "streaming.h"
#include "lbm_definitions.h"

void DoStreaming(double *collide_field, double *stream_field, int *flag_field, int xlength){
    int x,nx,y,ny,z,nz,i,step=xlength+2;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                if(flag_field[x+y*step+z*step*step]==FLUID){
                    for(i=0;i<Q_LBM;i++){
                        nx=x-LATTICEVELOCITIES[i][0];
                        ny=y-LATTICEVELOCITIES[i][1];
                        nz=z-LATTICEVELOCITIES[i][2];
                        
                        stream_field[Q_LBM*(x+y*step+z*step*step)+i]=
                                collide_field[Q_LBM*(nx+ny*step+nz*step*step)+i];
                    }
                }
            }
        }
    }
}