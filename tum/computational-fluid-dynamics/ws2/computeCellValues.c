#include "computeCellValues.h"
#include "LBDefinitions.h"
#include "helper.h"

void computeDensity(const double *const currentCell, double *density){
    /*TODO:if density is far from 1 something is going wrong */
    int i; *density=0;
    for(i=0;i<Q;i++)
        *density+=currentCell[i];
    
    if((*density-1.0)>EPS)
        ERROR("Density dropped below error tolerance.");
}

void computeVelocity(const double * const currentCell, const double * const density, double *velocity){
    int i;
    /* Indeces are hardcoded because of the possible performance gains and since 
     * we do not have alternating D */
    velocity[0]=0;
    velocity[1]=0;
    velocity[2]=0;
    
    for(i=0;i<Q;i++){
        velocity[0]+=currentCell[i]*LATTICEVELOCITIES[i][0];
        velocity[1]+=currentCell[i]*LATTICEVELOCITIES[i][1];
        velocity[2]+=currentCell[i]*LATTICEVELOCITIES[i][2];
    }
    
    velocity[0]/=*density;
    velocity[1]/=*density;
    velocity[2]/=*density;
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