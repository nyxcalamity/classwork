#ifndef _INITLB_H_
#define _INITLB_H_
#include "helper.h"
#include "LBDefinitions.h"

/* reads the parameters for the lid driven cavity scenario from a config file */
int readParameters(
    int *xlength,                       /* reads domain size. Parameter name: "xlength" */
    double *tau,                        /* relaxation parameter tau. Parameter name: "tau" */
    double *velocityWall,               /* velocity of the lid. Parameter name: "characteristicvelocity" */
    int *x_proc, int *y_proc, int *z_proc,          /* number of processes per axis */
    int *timesteps,                     /* number of timesteps. Parameter name: "timesteps" */
    int *timestepsPerPlotting,          /* timesteps between subsequent VTK plots. Parameter name: "vtkoutput" */
    int argc,                           /* number of arguments. Should equal 2 (program + name of config file */
    char *argv[]                        /* argv[1] shall contain the path to the config file */
);


/* initialises the particle distribution functions and the flagfield */
void initialiseFields(double *collideField, double *streamField,int *flagField, int xlength, 
        const int rank, const int number_of_ranks, const int x_proc);


/**
 * Function which returns the coordinate of process with given rank.
 * @param rank
 *          number of process
 * @param x_proc
 *          number of processes per axis
 */
int getRankX(const int rank, const int x_proc);
int getRankY(const int rank, const int x_proc);
int getRankZ(const int rank, const int x_proc);


/**
 * Function which fills up the array of neigbour_cells with neighbors of cell with given rank.
 * @param rank
 *          number of process
 * @param neigbour_cells
 *          array of neighbour cells which will be filled up.
 */
void findNeighborCells(const int rank, int *neighbor_cells, const int x_proc);


void initialiseBuffers(double **sendBuffer, double **readBuffer, const int x_sub_length);

#endif

