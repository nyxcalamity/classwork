#ifndef _LBDEFINITIONS_H_
#define _LBDEFINITIONS_H_

#define VERBOSE 0
#define MLUPS_CELLS_MIN 1000

#define D_LBM 3
#define Q_LBM 19
#define FLUID 0
#define NO_SLIP 1
#define MOVING_WALL 2
#define PARALLEL_BOUNDARY 9

#define EPS 0.05
#define SQRT3 1.73205080756887729

#define RIGHT_TO_LEFT 0
#define LEFT_TO_RIGHT 1
#define DOWN_TO_UP 2
#define UP_TO_DOWN 3
#define BACK_TO_FORTH 4
#define FORTH_TO_BACK 5

static const double C_S = 1.0/SQRT3;
static const double C_S_POW2_INV = 3.0;
static const double C_S_POW4_INV = 9.0;

static const int LATTICEVELOCITIES[19][3] = {
    {0,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{0,1,-1},{-1,-1,0},{0,-1,0},{1,-1,0},
    {-1,0,0}, {0,0,0},  {1,0,0}, {-1,1,0},{0,1,0}, {1,1,0},  {0,-1,1},{-1,0,1},
    {0,0,1},  {1,0,1},  {0,1,1}
};

static const double LATTICEWEIGHTS[19] = {
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 
    2.0/36.0, 12.0/36.0,2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 
    2.0/36.0, 1.0/36.0, 1.0/36.0
};

#define N_NORMAL 5
static const int NORMALVELOCITIES[6][5] = {
    { 1,  5,  8,  11, 15, },
    { 3,  7,  10, 13, 17, },
    { 4,  11, 12, 13, 18 },
    { 0,  5,  6,  7,  14, },
    { 14, 15, 16, 17, 18 },
    { 0,  1,  2,  3,  4 }};

#endif