#include "extraction_swap_injection.h"
#include "helper.h"

void extractionMPI(double *collideField, double *sendBuffer, const int direction, const int x_sub_length) {
    int x,y,z,i, step = x_sub_length+2;
    switch(direction) {
      case LEFT_TO_RIGHT:
        for(y=1;y<step-1;y++){
            for(z=1;z<step-1;z++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(z+step*y)+i]=
                            collideField[Q_LBM*(x_sub_length+y*step+z*step*step)+NORMALVELOCITIES[LEFT_TO_RIGHT][i]];
                }
            }
        }
        break;
      case RIGHT_TO_LEFT:
        for(y=1;y<step-1;y++){
            for(z=1;z<step-1;z++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(z+step*y)+i]=
                            collideField[Q_LBM*(1+y*step+z*step*step)+NORMALVELOCITIES[RIGHT_TO_LEFT][i]];
                }
            }
        }
        break;
      case DOWN_TO_UP:
        for(x=0;x<step;x++){
            for(z=1;z<step-1;z++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(x+step*z)+i]=
                            collideField[Q_LBM*(x+x_sub_length*step+z*step*step)+NORMALVELOCITIES[DOWN_TO_UP][i]];
                }
            }
        }
        break;
      case UP_TO_DOWN:
        for(x=0;x<step;x++){
            for(z=1;z<step-1;z++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(x+step*z)+i]=
                            collideField[Q_LBM*(x+1*step+z*step*step)+NORMALVELOCITIES[UP_TO_DOWN][i]];
                }
            }
        }
        break;
      case BACK_TO_FORTH:
        for(x=0;x<step;x++){
            for(y=0;y<step;y++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(x+step*y)+i]=
                            collideField[Q_LBM*(x+y*step+x_sub_length*step*step)+NORMALVELOCITIES[BACK_TO_FORTH][i]];
                }
            }
        }
        break;
      case FORTH_TO_BACK:
        for(x=0;x<step;x++){
            for(y=0;y<step;y++){
                for(i=0;i<N_NORMAL;i++){
                    sendBuffer[N_NORMAL*(x+step*y)+i]=
                            collideField[Q_LBM*(x+y*step+1*step*step)+NORMALVELOCITIES[FORTH_TO_BACK][i]];
                }
            }
        }
        break;
      default:
        ERROR("No such direction");
        break;
    }
}


void swapMPI(double *sendBuffer, double *readBuffer,const int * const neighbor_cells, 
        const int direction, const int x_sub_length){
    MPI_Status status;

    MPI_Send( sendBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
            neighbor_cells[direction], 1, MPI_COMM_WORLD);

    switch(direction) {
      case LEFT_TO_RIGHT:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction-1], 1, MPI_COMM_WORLD, &status);
        break;
      case RIGHT_TO_LEFT:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction+1], 1, MPI_COMM_WORLD, &status);
        break;
      case DOWN_TO_UP:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction+1], 1, MPI_COMM_WORLD, &status);
        break;
      case UP_TO_DOWN:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction-1], 1, MPI_COMM_WORLD, &status);
        break;
      case BACK_TO_FORTH:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction+1], 1, MPI_COMM_WORLD, &status);
        break;
      case FORTH_TO_BACK:
        MPI_Recv( readBuffer, N_NORMAL*(x_sub_length+2)*(x_sub_length+2), MPI_DOUBLE, 
                neighbor_cells[direction-1], 1, MPI_COMM_WORLD, &status);
        break;
      default:
        ERROR("No such direction");
        break;
    }
}


void injectionMPI(double *collideField, const int * const flagField, double *readBuffer, const int direction, const int x_sub_length) {
  int x,y,z,i, step = x_sub_length+2;
  switch(direction) {
    case LEFT_TO_RIGHT:
      for(y=1;y<step-1;y++){
          for(z=1;z<step-1;z++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[0+y*step+z*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*(0+y*step+z*step*step)+NORMALVELOCITIES[LEFT_TO_RIGHT][i]]=
                              readBuffer[N_NORMAL*(z+step*y)+i];
                  }
              }
          }
      }
      break;
    case RIGHT_TO_LEFT:
      for(y=1;y<step-1;y++){
          for(z=1;z<step-1;z++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[(x_sub_length+1)+y*step+z*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*((x_sub_length+1)+y*step+z*step*step)+NORMALVELOCITIES[RIGHT_TO_LEFT][i]]=
                              readBuffer[N_NORMAL*(z+step*y)+i];
                  }
              }
          }
      }
      break;
    case DOWN_TO_UP:
      for(x=0;x<step;x++){
          for(z=1;z<step-1;z++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[x+0*step+z*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*(x+0*step+z*step*step)+NORMALVELOCITIES[DOWN_TO_UP][i]]=
                              readBuffer[N_NORMAL*(x+step*z)+i];
                  }
              }
          }
      }
      break;
    case UP_TO_DOWN:
      for(x=0;x<step;x++){
          for(z=1;z<step-1;z++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[x+(x_sub_length+1)*step+z*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*(x+(x_sub_length+1)*step+z*step*step)+NORMALVELOCITIES[UP_TO_DOWN][i]]=
                              readBuffer[N_NORMAL*(x+step*z)+i];
                  }
              }
          }
      }
      break;
    case BACK_TO_FORTH:
      for(x=0;x<step;x++){
          for(y=0;y<step;y++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[x+y*step+0*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*(x+y*step+0*step*step)+NORMALVELOCITIES[BACK_TO_FORTH][i]]=
                              readBuffer[N_NORMAL*(x+step*y)+i];
                  }
              }
          }
      }
      break;
    case FORTH_TO_BACK:
      for(x=0;x<step;x++){
          for(y=0;y<step;y++){
              for(i=0;i<N_NORMAL;i++){
                  if(flagField[x+y*step+(x_sub_length+1)*step*step] == PARALLEL_BOUNDARY) {
                      collideField[Q_LBM*(x+y*step+(x_sub_length+1)*step*step)+NORMALVELOCITIES[FORTH_TO_BACK][i]]=
                              readBuffer[N_NORMAL*(x+step*y)+i];
                  }
              }
          }
      }
      break;
    default:
      ERROR("No such direction");
      break;
  }
}
