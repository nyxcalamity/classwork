#include "uvp.h"
#include "helper.h"

double du2dx(int i, int j, double **U, double dx, double alpha){
	double summand1 = pow((U[i][j]+U[i+1][j])/2, 2) - pow((U[i-1][j]+U[i][j])/2, 2);
	double summand2 = abs(U[i][j]+U[i+1][j])*(U[i][j]-U[i+1][j])/4 - abs(U[i-1][j]+U[i][j])*(U[i-1][j]-U[i][j])/4;
	return (summand1 + summand2*alpha)/dx;
}

double dv2dy(int i, int j, double **V, double dy, double alpha){
	double summand1 = pow((V[i][j]+V[i][j+1])/2, 2) - pow((V[i][j-1]+V[i][j])/2, 2);
	double summand2 = abs(V[i][j]+V[i][j+1])*(V[i][j]-V[i][j+1])/4 - abs(V[i][j-1]+V[i][j])*(V[i][j-1]-V[i][j])/4;
	return (summand1 + summand2*alpha)/dy;
}

double duvdy(int i, int j, double **U, double **V, double dy, double alpha){
	double summand1 = (V[i][j]+V[i+1][j])*(U[i][j]+U[i][j+1])/4 - (V[i][j-1]+V[i+1][j-1])*(U[i][j-1]+U[i][j])/4;
	double summand2 = abs(V[i][j]+V[i+1][j])*(U[i][j]-U[i][j+1])/4 - abs(V[i][j-1]+V[i+1][j-1])*(U[i][j-1]-U[i][j])/4;
	return (summand1+summand2*alpha)/dy;
}

double duvdx(int i, int j, double **U, double **V, double dx, double alpha){
	double summand1 = (U[i][j]+U[i][j+1])*(V[i][j]+V[i+1][j])/4 - (U[i-1][j]+U[i-1][j+1])*(V[i-1][j]+V[i][j])/4;
	double summand2 = abs(U[i][j]+U[i][j+1])*(V[i][j]-V[i+1][j])/4 - abs(U[i-1][j]+U[i-1][j+1])*(V[i-1][j]-V[i][j])/4;
	return (summand1+summand2*alpha)/dx;
}

double d2udx2(int i, int j, double **U, double dx){
	return (U[i+1][j]-2*U[i][j]+U[i-1][j])/pow(dx,2);
}

double d2vdx2(int i, int j, double **V, double dx){
	return (V[i+1][j]-2*V[i][j]+V[i-1][j])/pow(dx,2);
}

double d2udy2(int i, int j, double **U, double dy){
	return (U[i][j+1]-2*U[i][j]+U[i][j-1])/pow(dy,2);
}

double d2vdy2(int i, int j, double **V, double dy){
	return (V[i][j+1]-2*V[i][j]+V[i][j-1])/pow(dy,2);
}

void calculate_fg(
  double Re,
  double GX,
  double GY,
  double alpha,
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G
){
	/*TODO:for performance perform only 2 loops then fill the missing values*/
	int i,j;
	for(i=1;i<imax;i++){
		for(j=1;j<=jmax;j++){
			F[i][j] = U[i][j] +
					dt*((d2udx2(i,j,U,dx)+d2udy2(i,j,U,dy))/Re - du2dx(i,j,U,dx,alpha) - duvdy(i,j,U,V,dy,alpha) + GX);
		}
	}
	for(i=1;i<=imax;i++){
		for(j=1;j<jmax;j++){
			G[i][j] = V[i][j] +
					dt*((d2vdx2(i,j,V,dx)+d2vdy2(i,j,V,dy))/Re - dv2dy(i,j,V,dy,alpha) - duvdx(i,j,U,V,dx,alpha) + GY);
		}
	}
}

void calculate_rs(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **F,
  double **G,
  double **RS
){
	int i,j;
	for(i=1;i<=imax;i++){
		for(j=1;j<=jmax;j++){
			RS[i][j] = ((F[i][j]-F[i-1][j])/dx + (G[i][j]-G[i][j-1])/dy)/dt;
		}
	}
}

void calculate_dt(
  double Re,
  double tau,
  double *dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V
){
	/* TODO:move calculation to RAM */
	double c = Re/2 * 1/(1/pow(dx,2) + 1/pow(dy,2));
	double f = dx/maxAbs(U, imax, jmax+1);
	double l = dy/maxAbs(V, imax+1, jmax);
	*dt = tau * fmin(c, fmin(f,l));
}

void calculate_uv(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G,
  double **P
){
	/* TODO:again 4 loops, reduce to 2 */
	int i,j;
	for(i=1; i<imax; i++){
		for(j=1; j<=jmax;j++){
			U[i][j] = F[i][j] - (P[i+1][j]-P[i][j])*dt/dx;
		}
	}
	for(i=1; i<=imax; i++){
		for(j=1; j<jmax;j++){
			V[i][j] = G[i][j] - (P[i][j+1]-P[i][j])*dt/dy;
		}
	}
}
