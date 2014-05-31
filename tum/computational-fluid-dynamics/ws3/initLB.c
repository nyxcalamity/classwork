#include "initLB.h"
#include "helper.h"
#include <unistd.h>

int readParameters(int *rotate_pgm_coordinates,int *obstacle_start, int *obstacle_end, int ***pgm_data, 
        double *density_ref, double *density_in, int *boundaries, int *length, double *tau, 
        double *velocity_wall, int *timesteps, int *timesteps_per_plotting, int argc, char *argv[]){
    double *velocity_wall_1, *velocity_wall_2, *velocity_wall_3;
    int *xlength, *ylength, *zlength, *boundary_bottom, *boundary_top, *boundary_back, *boundary_front, 
            *boundary_left, *boundary_right, x, y;
    char pgm_file_name[50], pgm_file_name_full[60] = "", file_name[50] = "";
    
    strcat(file_name, "data/"); strcat(file_name, argv[1]); strcat(file_name, ".dat");
    printf("The data will be read from: %s\n", file_name);

    if(argc<2)
        ERROR("Not enough arguments. At least a path to init file is required.");
    if(access(file_name, R_OK) != 0)
        ERROR("Provided file path either doesn't exist or can not be read.");
    
    READ_INT(file_name, *rotate_pgm_coordinates);

    READ_STRING(file_name, pgm_file_name);
    strcat(pgm_file_name_full, "data/");
    strcat(pgm_file_name_full, pgm_file_name);
    strcat(pgm_file_name_full, ".pgm");
    printf("PGM data will be read from: %s\n", pgm_file_name_full);

    if(*rotate_pgm_coordinates) {
        xlength = &length[1];
        ylength = &length[0];
    } else {
        xlength = &length[0];
        ylength = &length[1];
    }
    zlength = &length[2];
    *pgm_data = read_pgm(pgm_file_name_full, xlength, ylength);

    /*
     * Check if our obstacle has correct shape(that no two cells are connected only with edge)
     */
    for(x=1; x<=*xlength; x++) {
        for(y=1; y<=*ylength; y++) {
            if( (*pgm_data)[x][y] == NO_SLIP ) {
                
                /* Check if there are any horizontal or diagonal neighbors */
                if( !((*pgm_data)[x+1][y]==NO_SLIP || (*pgm_data)[x-1][y]==NO_SLIP || 
                    (*pgm_data)[x][y+1]==NO_SLIP || (*pgm_data)[x][y-1]==NO_SLIP )
                    &&
                    ((*pgm_data)[x+1][y+1]==NO_SLIP || (*pgm_data)[x+1][y-1]==NO_SLIP || 
                    (*pgm_data)[x-1][y+1]==NO_SLIP || (*pgm_data)[x-1][y-1]==NO_SLIP) ){
                        printf("Problem with cell (%i,%i)", x, y);
                        ERROR("There some cells which are connected only with edges.");
                }
            }
        }
    }
    printf("PGM data is checked. Everything is fine!\n");
    READ_INT(file_name, *zlength);
    printf("xlength       = %i\n", *xlength);
    printf("ylength       = %i\n", *ylength);
    READ_INT(file_name, *obstacle_start);
    READ_INT(file_name, *obstacle_end);
    if(*obstacle_start > *zlength || *obstacle_start < 1)
    	ERROR("Obstacle start is outside of domain");
    if(*obstacle_end > *zlength || *obstacle_end < 1)
        ERROR("Obstacle end is outside of domain");
    if(*obstacle_start > *obstacle_end)
        ERROR("Wrong obstacle geometry: start > end");

    READ_DOUBLE(file_name, *density_ref);
    READ_DOUBLE(file_name, *density_in);
    READ_DOUBLE(file_name, *tau);

    boundary_bottom = &boundaries[0];
    boundary_top    = &boundaries[1];
    boundary_back   = &boundaries[2];
    boundary_front  = &boundaries[3];
    boundary_left   = &boundaries[4];
    boundary_right  = &boundaries[5];
    READ_INT(file_name, *boundary_bottom);
    READ_INT(file_name, *boundary_top);
    READ_INT(file_name, *boundary_back);
    READ_INT(file_name, *boundary_front);
    READ_INT(file_name, *boundary_left);
    READ_INT(file_name, *boundary_right);

    velocity_wall_1 = &velocity_wall[0];
    velocity_wall_2 = &velocity_wall[1];
    velocity_wall_3 = &velocity_wall[2];
    READ_DOUBLE(file_name, *velocity_wall_1);
    READ_DOUBLE(file_name, *velocity_wall_2);
    READ_DOUBLE(file_name, *velocity_wall_3);
    
    READ_INT(file_name, *timesteps);
    READ_INT(file_name, *timesteps_per_plotting);

    return 1;
}


void initialiseFields( const int rotate_pgm_coordinates, const int obstacle_start, 
        const int obstacle_end, int **pgm_data, int *boundaries, double *collide_field, 
        double *stream_field, int *flag_field, int *xlength){
    int x, y, z, i, step_x=xlength[0]+2, step_y=xlength[1]+2, step_z=xlength[2]+2;
    
    /* NOTE: We use z=xlength+1 as the moving wall */
    for(x=0;x<step_x;x++){
        for(y=0;y<step_y;y++){
            for(z=0;z<step_z;z++){
                /*
                 * The order of conditions is very important since edges are set to the first boundary.
                 * The edges are set to boundaries which have bigger priority in order not to let 
                 * FREE_SLIP be on the edge.
                 */
            	if (x==0 && y==0) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[4],boundaries[0]);
            	} else if (x==0 && y==step_y-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[4],boundaries[1]);
            	} else if (x==step_x-1 && y==0) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[5],boundaries[0]);
            	} else if (x==step_x-1 && y==step_y-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[5],boundaries[1]);
            	} else if (x==0 && z==0) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[4],boundaries[2]);
            	} else if (x==0 && z==step_z-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[4],boundaries[3]);
            	} else if (x==step_x-1 && z==0) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[5],boundaries[2]);
            	} else if (x==step_x-1 && z==step_z-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[5],boundaries[3]);
            	} else if (y==0 && z==0) {
                    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[0],boundaries[2]);
            	} else if (y==0 && z==step_z-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[0],boundaries[3]);
            	} else if (y==step_y-1 && z==0) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[1],boundaries[2]);
            	} else if (y==step_y-1 && z==step_z-1) {
            	    flag_field[x+y*step_x+z*step_x*step_y]=max(boundaries[1],boundaries[3]);
            	} else if(y == 0) {
                    flag_field[x+y*step_x+z*step_x*step_y]=boundaries[0];
            	} else if (y == step_y-1) {
                    flag_field[x+y*step_x+z*step_x*step_y]=boundaries[1];
            	} else if(z == 0) {
                    flag_field[x+y*step_x+z*step_x*step_y]=boundaries[2];
            	} else if(z == step_z-1) {
                	flag_field[x+y*step_x+z*step_x*step_y]=boundaries[3];
            	} else if(x == 0) {
                	flag_field[x+y*step_x+z*step_x*step_y]=boundaries[4];
            	} else if( x == step_x-1) {
                	flag_field[x+y*step_x+z*step_x*step_y]=boundaries[5];
            	} else if( (z >= obstacle_start) && (z <= obstacle_end ) ) {
            	    if(rotate_pgm_coordinates) {
            	        flag_field[x+y*step_x+z*step_x*step_y] = pgm_data[y][step_x-1 - x];
            	    } else {
            		flag_field[x+y*step_x+z*step_x*step_y] = pgm_data[x][y]; /* Obstacle shape. */
            	    }
            	} else {
                    flag_field[x+y*step_x+z*step_x*step_y]=FLUID;
            	}

                /* Initializing distributions for stream and collide fields */
                for(i=0;i<Q_LBM;i++){
                    /* 
                     * NOTE:Stream field is initialized to 0s because that helps to track down 
                     * mistakes and has no impact whatsoever to on the computation further on.
                     */
                    stream_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=0;
                    collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=LATTICEWEIGHTS[i];
                }
            }
        }
    }
    
    /* If obstacle cell is adjacent to FREE_SLIP boundary then we set this boundary to NO_SLIP */
    for(x=1;x<step_x-1;x++){
        for(y=1;y<step_y-1;y++){
            for(z=1;z<step_z-1;z++){
                if(flag_field[x+y*step_x+z*step_x*step_y]!=FLUID) {
                    /* Check if adjacent boundary cell is FREE_SLIP */
                    if(x+1 == step_x-1 && flag_field[(x+1)+y*step_x+z*step_x*step_y]==FREE_SLIP) {
                        flag_field[(x+1)+y*step_x+z*step_x*step_y]=NO_SLIP;
                    } else if(y+1 == step_y-1 && flag_field[x+(y+1)*step_x+z*step_x*step_y]==FREE_SLIP) {
                        flag_field[x+(y+1)*step_x+z*step_x*step_y]=NO_SLIP;
                    } else if(z+1 == step_z-1 && flag_field[x+y*step_x+(z+1)*step_x*step_y]==FREE_SLIP) {
                        flag_field[x+y*step_x+(z+1)*step_x*step_y]=NO_SLIP;
                    }
                    
                    if(x-1 == 0 && flag_field[(x-1)+y*step_x+z*step_x*step_y]==FREE_SLIP) {
                        flag_field[(x-1)+y*step_x+z*step_x*step_y]=NO_SLIP;
                    } else if(y-1 == 0 && flag_field[x+(y-1)*step_x+z*step_x*step_y]==FREE_SLIP) {
                        flag_field[x+(y-1)*step_x+z*step_x*step_y]=NO_SLIP;
                    } else if(z-1 == 0 && flag_field[x+y*step_x+(z-1)*step_x*step_y]==FREE_SLIP) {
                        flag_field[x+y*step_x+(z-1)*step_x*step_y]=NO_SLIP;
                    }
                }
            }
        }
    }
}
