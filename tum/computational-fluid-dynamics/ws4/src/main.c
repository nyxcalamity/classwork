#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"
#include <time.h>
#include <mpi.h>
#include "parallel.h"
#include "extraction_swap_injection.h"

/* Initializes MPI session and reads basic parameters */
void initializeMPI(int *rank, int *number_of_ranks, int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, number_of_ranks);
}

/* Finalizes MPI session */
void finalizeMPI(){
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

/* Validates the configured physical model by calculating characteristic numbers */
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
    if(mach_number >= 1)
        ERROR("Computed Mach number is too large.");
    if(reynolds_number > 500)
        ERROR("Computed Reynolds number is too large for simulation to be run on a laptop/pc.");
}

int main(int argc, char *argv[]){
    double *collideField=NULL, *streamField=NULL, num_cells, *swap = NULL, tau, velocityWall[3];
    int *flagField=NULL, xlength, timesteps, timestepsPerPlotting, rank, number_of_ranks, x_proc, 
            y_proc, z_proc, x_sub_length, b=0, t=0, mlups_exp=pow(10,6);
    /*
     * to distinguish to which process to send data
     * [0:left,1:rigth,2:top,3:bottom,4:front,5:back]
     */
    int neighbor_cells[6];
    /*
     * send and read buffers for all possible directions :
     * [0:left,1:rigth,2:top,3:bottom,4:front,5:back]
     */
    double *sendBuffer[6], *readBuffer[6];
    clock_t mlups_time;
    
    initializeMPI(&rank, &number_of_ranks, argc, argv);

    /* One process reads parameters and broadcasts them to others. */
    if(rank == 0) {
        readParameters(
            &xlength,&tau,velocityWall,&x_proc,&y_proc,&z_proc,&timesteps,
            &timestepsPerPlotting,argc,argv
        );
        x_sub_length = xlength/x_proc;
        validateModel(velocityWall, xlength, tau);
    } else {
        xlength=0; tau =0; velocityWall[0] = 0; velocityWall[1] = 0; velocityWall[2] = 0;
        x_proc = 0; y_proc = 0; z_proc = 0; timesteps = 0; timestepsPerPlotting = 0;
        x_sub_length = 0;
    }

    MPI_Bcast( &xlength, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( velocityWall, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( &x_proc, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &y_proc, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &z_proc, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &timestepsPerPlotting, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &x_sub_length, 1, MPI_INT, 0, MPI_COMM_WORLD );

    if(rank != 0 && VERBOSE)
        printf("rank=%i,xlength=%i,tau=%f,velocityWall=(%f,%f,%f)\nx_proc=%i,y_proc=%i,z_proc=%i,timesteps=%i,timestepsPerPlotting=%i\n\n",
            rank,xlength,tau,velocityWall[0],velocityWall[1],velocityWall[2],x_proc,y_proc,z_proc,timesteps,timestepsPerPlotting);

    if(number_of_ranks%(x_proc*y_proc*z_proc))
        ERROR("number_of_ranks is not enough to have x_proc, y_proc and z_proc processes on axis");

    num_cells = (x_sub_length+2)*(x_sub_length+2)*(x_sub_length+2);
    collideField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    streamField = malloc(Q_LBM*num_cells*sizeof(*collideField));
    flagField = malloc(num_cells*sizeof(*flagField));
    initialiseFields(collideField,streamField,flagField,x_sub_length,rank,number_of_ranks, x_proc);
    
    findNeighborCells(rank, neighbor_cells, x_proc);
    if(VERBOSE)
        printf("rank=%i,neighbor_cells:{%i,%i,%i,%i,%i,%i}\n",rank,neighbor_cells[0],neighbor_cells[1],
                neighbor_cells[2],neighbor_cells[3],neighbor_cells[4],neighbor_cells[5]);
    if(VERBOSE)
        writeFlagField(flagField,"img/flagField",x_sub_length, rank);

    for(b=0;b<6;b++) {
        sendBuffer[b] = malloc(N_NORMAL*(x_sub_length+2)*(x_sub_length+2)*sizeof(*sendBuffer[b]));
        readBuffer[b] = malloc(N_NORMAL*(x_sub_length+2)*(x_sub_length+2)*sizeof(*readBuffer[b]));
    }
    initialiseBuffers(sendBuffer, readBuffer, x_sub_length);

    if(VERBOSE) {
        writeBuffer(sendBuffer[FORTH_TO_BACK], "debug/sendBuffer[FORTH_TO_BACK]", 0, x_sub_length, rank);
        writeBuffer(readBuffer[FORTH_TO_BACK], "debug/readBuffer[FORTH_TO_BACK]", 0, x_sub_length, rank);
    }

    if(VERBOSE)
        writeField(collideField,"debug/collideField",t,x_sub_length, rank);

    for(t=0;t<=timesteps;t++){
        mlups_time = clock();

        extractionMPI(collideField, sendBuffer[LEFT_TO_RIGHT], LEFT_TO_RIGHT, x_sub_length);
        swapMPI(sendBuffer[LEFT_TO_RIGHT], readBuffer[LEFT_TO_RIGHT],neighbor_cells, LEFT_TO_RIGHT, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[LEFT_TO_RIGHT], LEFT_TO_RIGHT, x_sub_length);

        extractionMPI(collideField, sendBuffer[RIGHT_TO_LEFT], RIGHT_TO_LEFT, x_sub_length);
        swapMPI(sendBuffer[RIGHT_TO_LEFT], readBuffer[RIGHT_TO_LEFT],neighbor_cells, RIGHT_TO_LEFT, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[RIGHT_TO_LEFT], RIGHT_TO_LEFT, x_sub_length);

        extractionMPI(collideField, sendBuffer[DOWN_TO_UP], DOWN_TO_UP, x_sub_length);
        swapMPI(sendBuffer[DOWN_TO_UP], readBuffer[DOWN_TO_UP],neighbor_cells, DOWN_TO_UP, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[DOWN_TO_UP], DOWN_TO_UP, x_sub_length);

        extractionMPI(collideField, sendBuffer[UP_TO_DOWN], UP_TO_DOWN, x_sub_length);
        swapMPI(sendBuffer[UP_TO_DOWN], readBuffer[UP_TO_DOWN],neighbor_cells, UP_TO_DOWN, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[UP_TO_DOWN], UP_TO_DOWN, x_sub_length);

        extractionMPI(collideField, sendBuffer[BACK_TO_FORTH], BACK_TO_FORTH, x_sub_length);
        swapMPI(sendBuffer[BACK_TO_FORTH], readBuffer[BACK_TO_FORTH],neighbor_cells, BACK_TO_FORTH, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[BACK_TO_FORTH], BACK_TO_FORTH, x_sub_length);

        extractionMPI(collideField, sendBuffer[FORTH_TO_BACK], FORTH_TO_BACK, x_sub_length);
        swapMPI(sendBuffer[FORTH_TO_BACK], readBuffer[FORTH_TO_BACK],neighbor_cells, FORTH_TO_BACK, x_sub_length);
        injectionMPI(collideField, flagField, readBuffer[FORTH_TO_BACK], FORTH_TO_BACK, x_sub_length);

        /* Copy pdfs from neighbouring cells into collide field */
        doStreaming(collideField,streamField,flagField,x_sub_length);
        /* Perform the swapping of collide and stream fields */
        swap = collideField; collideField = streamField; streamField = swap;
        /* Compute post collision distributions */
        doCollision(collideField,flagField,&tau,x_sub_length);
        /* Treat boundaries */
        treatBoundary(collideField,flagField,velocityWall,x_sub_length);
        /* Print out the MLUPS value */
        mlups_time = clock()-mlups_time;
        if(num_cells > MLUPS_CELLS_MIN)
            printf("Process: #%d Time step: #%d MLUPS: %f\n", rank, t, 
                    num_cells/(mlups_exp*(double)mlups_time/CLOCKS_PER_SEC));
        /* Print out vtk output if needed */
        if (t%timestepsPerPlotting==0)
            writeVtkOutput(collideField,flagField,"img/lbm-img",t,x_sub_length, rank, x_proc);
    }

    /* Free memory */
    for(b=0;b<6;b++) {
        free(sendBuffer[b]);
        free(readBuffer[b]);
    }
    free(collideField);
    free(streamField);
    free(flagField);
    
    fflush(stdout);
    fflush(stderr);
    
    finalizeMPI();

    printf("Simulation complete for process %i.\n", rank);
    return 0;
}
#endif
