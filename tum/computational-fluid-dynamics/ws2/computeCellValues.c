#include "computeCellValues.h"
#include "LBDefinitions.h"

void computeDensity(const double *const currentCell, double *density){
    int i; *density=0;
    for(i=0;i<Q;i++)
        *density+=currentCell[i];
}

void computeVelocity(const double * const currentCell, const double * const density, double *velocity){
    int i,d;
    /*TODO:perhaps if we have D fixed and it will never change it makes more sence 
     to hardcode this part => saves operating memory and improves performance */
    for(d=0;d<D;d++)
        velocity[d]=0;
    
    for(i=0;i<Q;i++)
        for(d=0;d<D;d++)
            velocity[d]+=currentCell[i]*LATTICEVELOCITIES[i][d];
    
    for(d=0;d<D;d++)
        velocity[d]/=*density;
}

void computeFeq(const double * const density, const double * const velocity, double *feq){
    int i;
    double s1, s2, s3; /* summands */
    /* Indexes are hardcoded to improve program performance */
    for(i=0;i<Q;i++){
        s1 = LATTICEVELOCITIES[i][1]*velocity[1]+LATTICEVELOCITIES[i][2]*velocity[2]+
                LATTICEVELOCITIES[i][3]*velocity[3];
        s2 = s1*s1;
        s3 = velocity[1]*velocity[1]+velocity[2]*velocity[2]+velocity[3]*velocity[3];
        
        feq[i]=LATTICEWEIGHTS[i]*(*density)*(1+s1*C_S_POW2_INV+s2*C_S_POW4_INV/2.0-s3*C_S_POW2_INV/2.0);
    }
}