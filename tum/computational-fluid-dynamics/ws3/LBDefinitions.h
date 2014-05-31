#ifndef _LBDEFINITIONS_H_
#define _LBDEFINITIONS_H_

#define VERBOSE 0
#define OUTPUT_FLAGFIELD 0

#define D_LBM 3
#define Q_LBM 19

/* Types of boundary or fluid. BCs are set in priority required for proper treatment of edges. */
#define FLUID 0
#define FREE_SLIP 1
#define NO_SLIP 2
#define MOVING_WALL 3
#define INFLOW 4
#define OUTFLOW 5
#define PRESSURE_IN 6

#define EPS 0.1
/* Take from http://en.wikipedia.org/wiki/Square_root_of_3 */
#define SQRT3 1.73205080756887729353

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

static const double NORMALVELOCITIES16_2[5] = { 14, 15, 16, 17, 18, };
static const double MIRROREDVELOCITIES16_2[5] = { 0, 1, 2, 3, 4, };

static const double NORMALVELOCITIES12_6[5] = { 4, 11, 12, 13, 18, };
static const double MIRROREDVELOCITIES12_6[5] = { 0, 5, 6, 7, 14, };

static const double NORMALVELOCITIES10_8[5] = { 3, 7, 10, 13, 17, };
static const double MIRROREDVELOCITIES10_8[5] = { 1, 5, 8, 11, 15, };

#endif