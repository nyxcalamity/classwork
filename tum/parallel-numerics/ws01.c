/*
 * TASK: Write a simple application that defines a constant pi on the first node in a cluster and
 * afterwards sends this constant to all other nodes one by one. Use only the MPI operations 
 * MPI Send and MPI Recv.
 * 
 * @author Denys Sobchyshak
 */
#include <mpi.h>
#include <math.h>
#include <stdio.h>

int main (int argc, char * argv []) {
    int i, myrank, nproc;
    double pi;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD , &myrank);
    
    if (myrank == 0) {
        pi = M_PI;
        for (i=1; i<nproc; ++i) {
            MPI_Send(&pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    printf("Value of Pi on process %i: %f\n", myrank, pi);
    
    MPI_Finalize();
    return 0;
}