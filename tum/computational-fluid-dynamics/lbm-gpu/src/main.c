#ifndef _MAIN_C_
#define _MAIN_C_

#include <time.h>
#include "collision.h"
#include "streaming.h"
#include "initialization.h"
#include "visualization.h"
#include "boundary.h"
#include "collision_gpu.h"

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


/* Validates the configured physical model by calculating characteristic numbers */
void validateModel(double velocity_wall[3], int xlength, double tau){
    double u_wall_length,mach_number, reynolds_number;
    /* Compute Mach number and Reynolds number */
    u_wall_length=sqrt(velocity_wall[0]*velocity_wall[0]+velocity_wall[1]*velocity_wall[1]+
            velocity_wall[2]*velocity_wall[2]);
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
    double *collide_field=NULL, *stream_field=NULL, tau, velocity_wall[3], num_cells;
    int *flag_field=NULL, xlength, t, timesteps, timesteps_per_plotting;
    //clock_t mlups_time; double *swap=NULL; int mlups_exp=pow(10,6);
    size_t size;
    
    readParameters(&xlength,&tau,velocity_wall,&timesteps,&timesteps_per_plotting,argc,argv);
    validateModel(velocity_wall, xlength, tau);
    
    num_cells = pow(xlength+2, D_LBM);
    size = Q_LBM*num_cells*sizeof(double);
    collide_field = (double*)malloc(size);
    stream_field = (double*)malloc(size);
    flag_field = (int*)malloc(num_cells*sizeof(int));
    initialiseFields(collide_field,stream_field,flag_field,xlength);
    
    CudaTest(collide_field, size);

//    for(t=0;t<timesteps;t++){
//        mlups_time = clock();
        /* Copy pdfs from neighbouring cells into collide field */
//        doStreaming(collide_field,stream_field,flag_field,xlength);
        /* Perform the swapping of collide and stream fields */
//        swap = collide_field; collide_field = stream_field; stream_field = swap;
        /* Compute post collision distributions */
//        doCollision(collide_field,flag_field,&tau,xlength);
        /* Treat boundaries */
//        treatBoundary(collide_field,flag_field,velocity_wall,xlength);
        /* Print out the MLUPS value */
//        mlups_time = clock()-mlups_time;
//        if(VERBOSE && num_cells > MLUPS_MIN_CELLS)
//            printf("Time step: #%d MLUPS: %f\n", t,
//                    num_cells/(mlups_exp*(double)mlups_time/CLOCKS_PER_SEC));
        /* Print out vtk output if needed */
//        if (t%timesteps_per_plotting==0)
//            writeVtkOutput(collide_field,flag_field,"img/lbm-img",t,xlength);
        
//        if(VERBOSE)
//            printField(collide_field, xlength);
//    }

    t=0;
    writeVtkOutput(collide_field,flag_field,"img/lbm-img",t,xlength);

    /* Free memory */
    free(collide_field);
    free(stream_field);
    free(flag_field);
    
    printf("Simulation complete.\n");
    return 0;
}
#endif
