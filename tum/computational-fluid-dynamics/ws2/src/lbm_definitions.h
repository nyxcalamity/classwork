#ifndef _LBDEFINITIONS_H_
#define _LBDEFINITIONS_H_

#define VERBOSE 0

#define D_LBM 3
#define Q_LBM 19

#define FLUID 0
#define NO_SLIP 1
#define MOVING_WALL 2

#define EPS 0.05
#define SQRT3 1.73205080756887729

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
#endif