#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"
#include <time.h>
#include "mpi.h"

/**
 * Function that prints out the point by point values of the provided field (4D).
 * @param field
 *          linerized 4D array, with (x,y,z,i)=Q*(x+y*(ncell+2)+z*(ncell+2)*(ncell+2))+i
 * @param ncell
 *          number of inner cells, the ones which are there before adding a boundary layer
 */
void printField(double *field, int ncell){
    int x,y,z,i,step=ncell+2;
    
    for(x=0;x<step;x++){
        for(y=0;y<step;y++){
            for(z=0;z<step;z++){
                printf("(%d,%d,%d): ",x,y,z);
                for(i=0;i<Q_LBM;i++){
                    printf("%f ",field[Q_LBM*(x+y*step+z*step*step)+i]);
                }
                printf("\n");
            }
        }
    }
}


void initializeMPI(int *rank, int *rank_size, int *argc, char **argv[]){
    MPI_Init(argc,argv);
    MPI_Comm_size(MPI_COMM_WORLD, rank);
    MPI_Comm_rank(MPI_COMM_WORLD, rank_size);
}


void finalizeMPI(){
    MPI_Finalize();
}


void validateModel(double velocityWall[3], int xlength, double tau){
    double u_wall_length,mach_number, reynolds_number;
    /* Compute Mach number and Reynolds number */
    u_wall_length=sqrt(velocityWall[0]*velocityWall[0]+velocityWall[1]*velocityWall[1]+
            velocityWall[2]*velocityWall[2]);
    mach_number = u_wall_length*SQRT3;
    reynolds_number=u_wall_length*xlength*C_S_POW2_INV/(tau-0.5);
    printf("Computed Mach number: %f\n", mach_number);
    printf("Computed Reynolds number: %f\n", reynolds_number);
    
    /* Check if characteristic numbers are correct */
    if(mach_number >= 0.1)
        ERROR("Computed Mach number is too large.");
    if(reynolds_number > 500)
        ERROR("Computed Reynolds number is too large for simulation to be run on a laptop/pc.");
}


int main(int argc, char *argv[]){
    double *collideField=NULL, *streamField=NULL, *swap=NULL, tau, velocityWall[3], num_cells;
    int *flagField=NULL, xlength, t, timesteps, timestepsPerPlotting, mlups_exp=pow(10,6), 
            x_proc, y_proc, z_proc;
    clock_t mlups_time;
    
    readParameters(&xlength,&tau,velocityWall,&timesteps,&timestepsPerPlotting,argc,argv,&x_proc,
            &y_proc,&z_proc);
    validateModel(velocityWall, xlength, tau);

    num_cells = pow(xlength+2, D_LBM);
    collideField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    streamField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    flagField = malloc(num_cells*sizeof(*flagField));
    initialiseFields(collideField,streamField,flagField,xlength);
    
    for(t=0;t<timesteps;t++){
        mlups_time = clock();
        /* Copy pdfs from neighbouring cells into collide field */
        doStreaming(collideField,streamField,flagField,xlength);
        /* Perform the swapping of collide and stream fields */
        swap = collideField; collideField = streamField; streamField = swap;
        /* Compute post collision distributions */
        doCollision(collideField,flagField,&tau,xlength);
        /* Treat boundaries */
        treatBoundary(collideField,flagField,velocityWall,xlength);
        /* Print out the MLUPS value */
        mlups_time = clock()-mlups_time;
        if(num_cells >= MLUPS_CELLS_MIN)
            printf("Time step: #%d MLUPS: %f\n", t, num_cells/(mlups_exp*(double)mlups_time/CLOCKS_PER_SEC));
        /* Print out vtk output if needed */
        if (t%timestepsPerPlotting==0)
            writeVtkOutput(collideField,flagField,"img/lbm-img",t,xlength);
        if(VERBOSE)
            printField(collideField, xlength);
    }
    
    /* Free memory */
    free(collideField);
    free(streamField);
    free(flagField);
    
    printf("Simulation complete.");
    return 0;
}
#endif