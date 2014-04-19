#include "helper.h"
#include "visual.h"
#include "init.h"
#include "uvp.h"
#include "boundary_val.h"
#include <stdio.h>


/**
 * The main operation reads the configuration file, initializes the scenario and
 * contains the main loop. So here are the individual steps of the algorithm:
 *
 * - read the program configuration file using read_parameters()
 * - set up the matrices (arrays) needed using the matrix() command
 * - create the initial setup init_uvp(), init_flag(), output_uvp()
 * - perform the main loop
 * - trailer: destroy memory allocated and do some statistics
 *
 * The layout of the grid is described by the first figure below, the enumeration
 * of the whole grid is given by the second figure. All the unknowns correspond
 * to a two dimensional degree of freedom layout, so they are not stored in
 * arrays, but in a matrix.
 *
 * @image html grid.jpg
 *
 * @image html whole-grid.jpg
 *
 * Within the main loop the following big steps are done (for some of the 
 * operations a definition is defined already within uvp.h):
 *
 * - calculate_dt() Determine the maximal time step size.
 * - boundaryvalues() Set the boundary values for the next time step.
 * - calculate_fg() Determine the values of F and G (diffusion and confection).
 *   This is the right hand side of the pressure equation and used later on for
 *   the time step transition.
 * - calculate_rs()
 * - Iterate the pressure poisson equation until the residual becomes smaller
 *   than eps or the maximal number of iterations is performed. Within the
 *   iteration loop the operation sor() is used.
 * - calculate_uv() Calculate the velocity at the next time step.
 */
int main(int argn, char** args){
	/* Read the problem parameters */
	double Re,UI,VI,PI,GX,GY, 				/* problem dependent quantities */
		xlength,ylength,dx,dy,			 	/* geometry data */
		t,t_end,dt,tau,dt_value,			/* time stepping data */
		alpha,omg,eps,res; 					/* pressure iteration data */
	int itermax, n, imax, jmax;				/* max iterations, iteration step and # of interior cells */
	read_parameters("cavity100.dat", &Re, &UI, &VI, &PI, &GX, &GY, &t_end, &xlength, &ylength, &dt, &dx, &dy,
			&imax, &jmax, &alpha, &omg, &tau, &itermax, &eps, &dt_value);

	/* Set t := 0, n := 0 */
	t = 0; n = 0;

	/*	Assign initial values to u, v, p */
	double **U = matrix(0, imax, 0, jmax+1);
	double **F = matrix(0, imax, 0, jmax+1);
	double **V = matrix(0, imax+1, 0, jmax);
	double **G = matrix(0, imax+1, 0, jmax);
	double **P = matrix(0, imax+1, 0, jmax+1);
	double **RS= matrix(0, imax+1, 0, jmax+1);
	init_uvp(UI, VI, PI, imax, jmax, U, V, P);

	while (t < t_end){
		/* Select δt according to (14) */
		calculate_dt(Re, tau, &dt, dx, dy, imax, jmax, U, V);

		/* Set boundary values for u and v according to (15),(16) */
		boundaryvalues(imax, jmax, U, V);

		/* Compute F (n) and G (n) according to (10),(11),(18) */
		calculate_fg(Re, GX, GY, alpha, dt, dx, dy, imax, jmax, U, V, F, G);

		/* Compute the right-hand side rs of the pressure equation (12) */
		calculate_rs(dt, dx, dy, imax, jmax, F, G, RS);

		int it = 0;
		while (it < itermax && res > eps){
			/*
			//	TODO:Perform a SOR iteration according to (19) using the
			//	TODO:provided function and retrieve the residual res
			*/
			it++;
		}

		/* Compute u (n+1) and v (n+1) according to (8),(9) */
		calculate_uv(dt,dx,dy,imax,jmax,U,V,F,G,P);

		/*TODO:Output of u, v, p values for visualization, if necessary */

		t+=dt; n++;
	}

	/* TODO:Output of u, v, p for visualization */

	/* Freeing memory */
	free_matrix(U, 0, imax, 0, jmax+1);
	free_matrix(F, 0, imax, 0, jmax+1);
	free_matrix(V, 0, imax+1, 0, jmax);
	free_matrix(G, 0, imax+1, 0, jmax);
	free_matrix(P, 0, imax+1, 0, jmax+1);
	free_matrix(RS, 0, imax+1, 0, jmax+1);

	return -1;
}
