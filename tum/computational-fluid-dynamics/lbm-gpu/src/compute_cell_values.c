#include "compute_cell_values.h"
#include "lbm_definitions.h"
#include "helper.h"

void ComputeDensity(const double *const current_cell, double *density){
    int i; *density=0;
    for(i=0;i<Q_LBM;i++)
        *density+=current_cell[i];

    /* Density should be close to a unit (ρ~1) */
    if((*density-1.0)>EPS)
        ERROR("Density dropped below error tolerance.");
}


void ComputeVelocity(const double * const current_cell, const double * const density, 
        double *velocity){
    int i;
    /* NOTE:Indeces are hardcoded because of the possible performance gains and since 
     * we do not have alternating D */
    velocity[0]=0;
    velocity[1]=0;
    velocity[2]=0;
    
    for(i=0;i<Q_LBM;i++){
        velocity[0]+=current_cell[i]*LATTICE_VELOCITIES[i][0];
        velocity[1]+=current_cell[i]*LATTICE_VELOCITIES[i][1];
        velocity[2]+=current_cell[i]*LATTICE_VELOCITIES[i][2];
    }
    
    velocity[0]/=*density;
    velocity[1]/=*density;
    velocity[2]/=*density;
}


void ComputeFeq(const double * const density, const double * const velocity, double *feq){
    int i;
    double s1, s2, s3; /* summands */
    /* NOTE:Indexes are hardcoded to improve program performance */
    for(i=0;i<Q_LBM;i++){
        s1 = LATTICE_VELOCITIES[i][0]*velocity[0]+LATTICE_VELOCITIES[i][1]*velocity[1]+
                LATTICE_VELOCITIES[i][2]*velocity[2];
        s2 = s1*s1;
        s3 = velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2];
        
        feq[i]=LATTICE_WEIGHTS[i]*(*density)*(1+s1*C_S_POW2_INV+s2*C_S_POW4_INV/2.0-s3*C_S_POW2_INV/2.0);
        
        /* Probability distribution function can not be less than 0 */
        if (feq[i] < 0)
            ERROR("Probability distribution function can not be negative.");
    }
}
