#include "initLB.h"
#include "helper.h"
#include <unistd.h>


/* [0:left,1:rigth,2:top,3:bottom,4:front,5:back] */
void findNeighborCells(const int rank, int *neighbor_cells, const int x_proc) {
    neighbor_cells[0] = rank%x_proc ? rank-1: MPI_PROC_NULL;
    neighbor_cells[1] = (rank+1)%x_proc ? rank+1: MPI_PROC_NULL;
    neighbor_cells[2] = (getRankY(rank,x_proc)+1)%x_proc ? rank+x_proc : MPI_PROC_NULL;
    neighbor_cells[3] = (getRankY(rank,x_proc))%x_proc ? rank-x_proc : MPI_PROC_NULL;
    neighbor_cells[4] = (getRankZ(rank,x_proc)+1)%x_proc ? rank+x_proc*x_proc : MPI_PROC_NULL;
    neighbor_cells[5] = (getRankZ(rank,x_proc))%x_proc ? rank-x_proc*x_proc : MPI_PROC_NULL;
}

int readParameters(int *xlength, double *tau, double *velocityWall, 
        int *x_proc, int *y_proc, int *z_proc,
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

    READ_INT(argv[1], *x_proc);
    READ_INT(argv[1], *y_proc);
    READ_INT(argv[1], *z_proc);
    if(*xlength%*x_proc || *xlength%*y_proc || *xlength%*z_proc)
        ERROR("With the given number of processes per axis x_proc, y_proc, z_proc the domain of the given size xlength can't be divided in equal number of cells per process.");
    if(*x_proc!=*y_proc || *x_proc!=*z_proc || *y_proc!=*z_proc)
        ERROR("x_proc, y_proc and z_proc should have the same size");

    READ_INT(argv[1], *timesteps);
    READ_INT(argv[1], *timestepsPerPlotting);
    
    return 1;
}

int getRankX(const int rank, const int x_proc) {
    return rank%x_proc;
}
int getRankY(const int rank, const int x_proc) {
    return (rank/x_proc)%x_proc;
}
int getRankZ(const int rank, const int x_proc) {
    return rank/(x_proc*x_proc);
}

void initialiseFields(double *collideField, double *streamField, int *flagField, int xlength, const int rank, const int number_of_ranks, const int x_proc){
    int x,y,z,i,step=xlength+2;
    
    /* NOTE: We use z=xlength+1 as the moving wall */
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                /* Initializing flags */
                if(y == xlength+1 && getRankY(rank,x_proc)==x_proc-1)
                    flagField[x+y*step+z*step*step]=MOVING_WALL;
                else if((x == 0 && getRankX(rank,x_proc)==0) || (x == xlength+1 && getRankX(rank,x_proc)==x_proc-1)
                    || (y == 0 && getRankY(rank,x_proc)==0)
                    || (z == xlength+1 && getRankZ(rank,x_proc)==x_proc-1) || (z == 0  && getRankZ(rank,x_proc)==0))
                    flagField[x+y*step+z*step*step]=NO_SLIP;
                else if(x == 0 || x == xlength+1 || y == 0 || y == xlength+1 || z == 0 || z == xlength+1)
                    flagField[x+y*step+z*step*step]=PARALLEL_BOUNDARY;
                else
                    flagField[x+y*step+z*step*step]=FLUID;
                
                /* Initializing distributions for stream and collide fields */
                for(i=0;i<Q_LBM;i++){
                    /* NOTE:Stream field is initilized to 0s because that helps to 
                     * track down mistakes and has no impact whatsoever to on the 
                     * computation further on.
                     */
                    if(flagField[x+y*step+z*step*step] !=PARALLEL_BOUNDARY) {
                        streamField[Q_LBM*(x+y*step+z*step*step)+i]=0;
                        collideField[Q_LBM*(x+y*step+z*step*step)+i]=LATTICEWEIGHTS[i];
                    } else {
                        collideField[Q_LBM*(x+y*step+z*step*step)+i]=0;
                    }
                }
            }
        }
    }
}


void initialiseBuffers(double **sendBuffer, double **readBuffer, const int x_sub_length) {
    int x, y,b,i,step = x_sub_length+2;
    for(b=0; b<6; b++) {
        for(i=0; i<N_NORMAL; i++) {
            for(x=0; x<step; x++) {
                for(y=0; y<step; y++) {
                    sendBuffer[b][N_NORMAL*(x+y*step) + i] = 0;
                    readBuffer[b][N_NORMAL*(x+y*step) + i] = 0;
                }
            }
        }
    }
}
