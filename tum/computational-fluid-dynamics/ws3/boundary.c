#include "helper.h"
#include "boundary.h"
#include "LBDefinitions.h"
#include "computeCellValues.h"

/**
 * Inverts the value of the lattice index in order to find the vector opposite to the provided one.
 * @param i
 *      index to inverse
 * @return 
 *      inversed index
 */
int inv(int i){
    return (Q_LBM-1)-i;
}

int isVelociryPerpendicular(const int i) {
	return (LATTICEVELOCITIES[i][0]*LATTICEVELOCITIES[i][0] +
                LATTICEVELOCITIES[i][1]*LATTICEVELOCITIES[i][1] +
                LATTICEVELOCITIES[i][2]*LATTICEVELOCITIES[i][2]) == 1;
}

void findMirroredVelocities(const int respect_idx, int *normal_velocities, int *mirrored_velocities) {
    int index_of_save=0;

    /* Check if our code did some error. */
    if(!isVelociryPerpendicular(respect_idx))
    	ERROR("Wrong respective index");

    switch(respect_idx) {
        case 16:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = NORMALVELOCITIES16_2[index_of_save];
                mirrored_velocities[index_of_save] = MIRROREDVELOCITIES16_2[index_of_save];
            }
            break;
        case 2:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = MIRROREDVELOCITIES16_2[index_of_save];
                mirrored_velocities[index_of_save] = NORMALVELOCITIES16_2[index_of_save];
            }
            break;
        case 12:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = MIRROREDVELOCITIES12_6[index_of_save];
                mirrored_velocities[index_of_save] = NORMALVELOCITIES12_6[index_of_save];
            }
            break;
        case 6:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = NORMALVELOCITIES12_6[index_of_save];
                mirrored_velocities[index_of_save] = MIRROREDVELOCITIES12_6[index_of_save];
            }
            break;
        case 10:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = NORMALVELOCITIES10_8[index_of_save];
                mirrored_velocities[index_of_save] = MIRROREDVELOCITIES10_8[index_of_save];
            }
            break;
        case 8:
            for(index_of_save=0; index_of_save<5; index_of_save++) {
                normal_velocities[index_of_save] = MIRROREDVELOCITIES10_8[index_of_save];
                mirrored_velocities[index_of_save] = NORMALVELOCITIES10_8[index_of_save];
            }
            break;
        default:
            ERROR("Something went wrong in findMirroredVelocities() with velocity index");
            break;
    }
}

void treatBoundary(double *collide_field, int* flag_field, const double * const wall_velocity,
		           int *xlength, const double density_ref, const double density_in){
    int x, nx, y, ny, z, nz, i, step_x=xlength[0]+2, step_y=xlength[1]+2, step_z=xlength[2]+2;
    double density, dot_prod, velocity[3], feq[Q_LBM], *neighbor_cell;
    /* Used to save mirrored cells */
    int m, normal_velocities[5], mirrored_velocities[5];

    for(x=0;x<step_x;x++){
        for(y=0;y<step_y;y++){
            for(z=0;z<step_z;z++){
                if(flag_field[x+y*step_x+z*step_x*step_y]!=FLUID){
                    for(i=0;i<Q_LBM;i++){
                        nx=x+LATTICEVELOCITIES[i][0];
                        ny=y+LATTICEVELOCITIES[i][1];
                        nz=z+LATTICEVELOCITIES[i][2];

                        /* We don't need values outside of our extended domain */
                        if( 0<nx && nx<step_x-1 && 0<ny && ny<step_y-1 && 0<nz && nz<step_z-1
                        	&& flag_field[nx+ny*step_x+nz*step_x*step_y]==FLUID ){
                            if (flag_field[x+y*step_x+z*step_x*step_y]==MOVING_WALL){
                                /* Compute density in the neighbor cell */
                                computeDensity(&collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)],&density);
                                /* Compute dot product */
                                dot_prod=LATTICEVELOCITIES[i][0]*wall_velocity[0]+
                                        LATTICEVELOCITIES[i][1]*wall_velocity[1]+
                                        LATTICEVELOCITIES[i][2]*wall_velocity[2];
                                /* Assign the boundary cell value */
                                collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=
                                        collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+inv(i)]+
                                        2*LATTICEWEIGHTS[i]*density*C_S_POW2_INV*dot_prod;
                            } else if(flag_field[x+y*step_x+z*step_x*step_y]==NO_SLIP){
                                collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=
                                        collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+inv(i)];
                            } else if(flag_field[x+y*step_x+z*step_x*step_y]==OUTFLOW) {
                            	/*
                            	 * We calculate the velocity of the fluid in the neighbor cell
                            	 * after computing of the density of this(neighbor) cell.
                            	 */
                            	neighbor_cell=&collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)];
                            	computeDensity(neighbor_cell,&density);
                            	computeVelocity(neighbor_cell,&density,velocity);
                            	computeFeq(&density_ref,velocity,feq);
                                
                            	collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=
                            			feq[i] + feq[inv(i)] -
                            			collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+inv(i)];
                            } else if(flag_field[x+y*step_x+z*step_x*step_y]==INFLOW) {
                            	computeFeq(&density_ref,wall_velocity,feq);
                            	collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i] = feq[i];
                            } else if(flag_field[x+y*step_x+z*step_x*step_y]==FREE_SLIP) {
                            	if( isVelociryPerpendicular(i) ) {
                            		findMirroredVelocities(i, normal_velocities, mirrored_velocities);

                            		for(m=0; m<5; m++) {
                            			collide_field[ Q_LBM*(x+y*step_x+z*step_x*step_y)+normal_velocities[m] ]=
                            			    collide_field[ Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+mirrored_velocities[m] ];
                            		}
                            	}
                            } else if(flag_field[x+y*step_x+z*step_x*step_y]==PRESSURE_IN) {
                            	neighbor_cell=&collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)];
                            	computeDensity(neighbor_cell,&density);
                            	computeVelocity(neighbor_cell,&density,velocity);
                            	computeFeq(&density_in,velocity,feq);

                            	collide_field[Q_LBM*(x+y*step_x+z*step_x*step_y)+i]=
                            			feq[i] + feq[inv(i)] -
                            			collide_field[Q_LBM*(nx+ny*step_x+nz*step_x*step_y)+inv(i)];
                            }
                        }
                    }
                }
            }
        }
    }
}