#ifndef _INITLB_H_
#define _INITLB_H_

/** reads the parameters for the lid driven cavity scenario from a config file */
int ReadParameters(
    int *xlength,                       /* reads domain size. Parameter name: "xlength" */
    double *tau,                        /* relaxation parameter tau. Parameter name: "tau" */
    double *velocity_wall,              /* velocity of the lid. Parameter name: "characteristicvelocity" */
    int *timesteps,                     /* number of timesteps. Parameter name: "timesteps" */
    int *timesteps_per_plotting,        /* timesteps between subsequent VTK plots. Parameter name: "vtkoutput" */
    int argc,                           /* number of arguments. Should equal 2 (program + name of config file */
    char *argv[]                        /* argv[1] shall contain the path to the config file */
);


/* initialises the particle distribution functions and the flagfield */
void InitialiseFields(double *collide_field, double *stream_field,int *flag_field, int xlength);
#endif
