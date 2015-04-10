/*
 * @author Denys Sobchyshak
 */
#include <mpi.h>
#include "stdio.h"
#include <stdlib.h>

#define LENGTH 10000000


int main (int argc, char * argv []) {
    //Variables
    int i, myrank, nproc;
    double start_time, end_time, min_start_time, max_end_time;
    MPI_Status status;
    float *a, *b, *r, *a_loc, *b_loc, *r_loc;
    
    // Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD , &myrank);
    start_time = MPI_Wtime();
    
    int chunk_size = LENGTH/nproc;
    if (myrank == 0) {
        a = (float*) malloc(LENGTH*sizeof(float));
        b = (float*) malloc(LENGTH*sizeof(float));
        r = (float*) malloc(LENGTH*sizeof(float));
        
        for (i=0; i<LENGTH; ++i) {
            a[i] = rand()%100;
            b[i] = rand()%100;
        }
    }
    a_loc = (float*) malloc(chunk_size*sizeof(float));
    b_loc = (float*) malloc(chunk_size*sizeof(float));
    r_loc = (float*) malloc(chunk_size*sizeof(float));
    
    MPI_Scatter(a, chunk_size, MPI_FLOAT, a_loc, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, chunk_size, MPI_FLOAT, b_loc, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    for (i=0; i<chunk_size; ++i) {
        r_loc[i] = a_loc[i]+b_loc[i];
    }
    
    MPI_Gather(r_loc, chunk_size, MPI_FLOAT, r, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (myrank == 0) {
        for (i=0; i<LENGTH; ++i) {
            if (r[i]-a[i]-b[i] != 0) {
                printf("%f+%f=%f\n", a[i], b[i], r[i]);
            }
        }
    }

    end_time = MPI_Wtime();
    //find min and max times
    MPI_Reduce(&start_time, &min_start_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_time, &max_end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //print status message
    if (myrank == 0) {
        printf("Elapsed time (secs): %f\n", max_end_time-min_start_time);
    }
    
    //Finalization
    MPI_Finalize();
    return 0;
}