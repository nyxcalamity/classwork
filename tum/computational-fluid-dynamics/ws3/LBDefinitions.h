#ifndef _LBDEFINITIONS_H_
#define _LBDEFINITIONS_H_

#define VERBOSE 0

#define D 3
#define Q 19
#define FLUID 0
#define NO_SLIP 1
#define MOVING_WALL 2
#define EPS 0.05
/* Take from http://en.wikipedia.org/wiki/Square_root_of_3 */
#define SQRT3 1.73205080756887729352744634150587236694280525381038062805580

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