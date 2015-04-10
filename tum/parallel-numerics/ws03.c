/*
 * @author Denys Sobchyshak
 */
#include <mpi.h>
#include "stdio.h"


double f(double x) {
    return -3*x*x+3;
}


int main (int argc, char * argv []) {
    int i, myrank, nproc;
    double interval, a, b, h, partial_sum=0, sum, buffer;
    double A=-1, B=1, n=4;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD , &myrank);
    
    interval = (B-A)/nproc;
    a = A+myrank*interval;
    b = (myrank == (nproc-1)) ? B : a+interval;
    h = (b-a)/n;
    
    for(i=0; i<n-1; i++) {
        partial_sum = f(a+i*h);
    }
    
    partial_sum = partial_sum*h+(f(a)+f(b))*h/2;
    printf("Computed partial integer on #%d: %f\n", myrank, partial_sum);
    
    if (myrank == 0) {
        sum = 0;
        for (i=1; i<nproc; ++i) {
            MPI_Recv(&buffer, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            sum += buffer;
        }
        printf("Computed integer: %f\n", sum);
    } else {
        MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}

