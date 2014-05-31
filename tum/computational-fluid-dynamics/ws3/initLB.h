#ifndef _INITLB_H_
#define _INITLB_H_
#include "helper.h"
#include "LBDefinitions.h"

/* reads the parameters for the lid driven cavity scenario from a config file */
int readParameters(
    int *rotatePGMCoordinates,          /* To know if rotation of coordinates from PGM file is needed */
    int *obstacleStart,                 /* Used to set up thikness of obstacle in z direction */
    int *obstacleEnd,                   /* Used to set up thikness of obstacle in z direction */
    int ***pgmData,                     /* How our obstacle looks like */
    double *densityRef,                 /* ρ(ref) */
    double *densityIn,                  /* ρ(in) */
    int *boundaries,                    /* reads boundaries type for each wall */
    int *xlength,                       /* reads domain size. Parameter name: "xlength" */
    double *tau,                        /* relaxation parameter tau. Parameter name: "tau" */
    double *velocityWall,               /* velocity of the lid. Parameter name: "characteristicvelocity" */
    int *timesteps,                     /* number of timesteps. Parameter name: "timesteps" */
    int *timestepsPerPlotting,          /* timesteps between subsequent VTK plots. Parameter name: "vtkoutput" */
    int argc,                           /* number of arguments. Should equal 2 (program + name of config file */
    char *argv[]                        /* argv[1] shall contain the path to the config file */
);


/* initialises the particle distribution functions and the flagfield */
void initialiseFields(
    const int rotatePGMCoordinates,
    const int obstacleStart,
    const int obstacleEnd,
    int **pgmData,
    int *boundaries,
    double *collideField,
    double *streamField,
    int *flagField,
    int *xlength
);

#endif

