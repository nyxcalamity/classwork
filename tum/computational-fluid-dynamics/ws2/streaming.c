#include "streaming.h"
#include "LBDefinitions.h"

void doStreaming(double *collideField, double *streamField, int *flagField, int xlength){
    int x,nx,y,ny,z,nz,i,step=xlength+2;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                if(flagField[x+y*step+z*step*step]==FLUID){
                    for(i=0;i<Q;i++){
                        nx=x-LATTICEVELOCITIES[i][0];
                        ny=y-LATTICEVELOCITIES[i][1];
                        nz=z-LATTICEVELOCITIES[i][2];
                        
                        streamField[Q*(x+y*step+z*step*step)+i]=
                                collideField[Q*(nx+ny*step+nz*step*step)+i];
                    }
                }
            }
        }
    }
}

