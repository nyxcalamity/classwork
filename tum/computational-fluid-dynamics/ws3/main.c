#ifndef _MAIN_C_
#define _MAIN_C_

#include "collision.h"
#include "streaming.h"
#include "initLB.h"
#include "visualLB.h"
#include "boundary.h"
#include <time.h>

int main(int argc, char *argv[]){
    double *collide_field=NULL, *stream_field=NULL, *swap=NULL, tau, velocity_wall[3],
            density_ref, density_in;
    int *flag_field=NULL, xlength[3], t, timesteps, timesteps_per_plotting, mlups_exp=pow(10,6), 
            num_cells, **pgm_data=NULL, obstacle_start, obstacle_end, rotate_pgm_coordinates;
    clock_t mlups_time;
    /*
     * boundaries[0] = bottom;  boundaries[1] = top;    boundaries[2] = back;
     * boundaries[3] = front;   boundaries[4] = left;   boundaries[5] = right;
     */
    int boundaries[6];

    readParameters(&rotate_pgm_coordinates, &obstacle_start, &obstacle_end, &pgm_data, &density_ref,
            &density_in, boundaries, xlength, &tau, velocity_wall, &timesteps, &timesteps_per_plotting,
            argc, argv);
    
    num_cells = (xlength[0]+2)*(xlength[1]+2)*(xlength[2]+2);
    collide_field = malloc(Q_LBM*num_cells*sizeof(*collide_field));
    stream_field = malloc(Q_LBM*num_cells*sizeof(*collide_field));
    flag_field = malloc(num_cells*sizeof(*flag_field));
    
    initialiseFields(rotate_pgm_coordinates, obstacle_start, obstacle_end, pgm_data, boundaries,
            collide_field, stream_field, flag_field, xlength);

    for(t=0;t<=timesteps;t++){
        mlups_time = clock();
        /* Copy pdfs from neighbouring cells into collide field */
        doStreaming(collide_field, stream_field, flag_field, xlength);
        /* Perform the swapping of collide and stream fields */
        swap = collide_field; collide_field = stream_field; stream_field = swap;
        /* Compute post collision distributions */
        doCollision(collide_field, flag_field, &tau, xlength);
        /* Treat boundaries */
        treatBoundary(collide_field, flag_field, velocity_wall,xlength, density_ref, density_in);
        /* Print out the MLUPS value */
        mlups_time = clock()-mlups_time;
        if(num_cells > MLUPS_THRESHOLD)
            printf("Time step: #%d MLUPS: %f\n", t, num_cells/(mlups_exp*(double)mlups_time/CLOCKS_PER_SEC));
        /* Print out vtk output if needed */
        if (t%timesteps_per_plotting==0)
            writeVtkOutput(collide_field, flag_field, "img/lbm-img", t, xlength);
        /* Output for debugging */
        if(VERBOSE){
            printField(collide_field, xlength);
            printFlagField(flag_field, xlength);
        }
    }

    /* Free memory */
    free(collide_field);
    free(stream_field);
    free(flag_field);
    free_imatrix(pgm_data,0,xlength[0]+2,0,xlength[1]+2);

    printf("Simulation complete.");
    return 0;
}
#endif