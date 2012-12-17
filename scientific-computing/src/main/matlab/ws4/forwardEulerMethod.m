function [ T ] = forwardEulerMethod( Nx,Ny,dt,t )
%forwardEulerMethod Calculates temperatures at the next timestep
%   Solves a system of ODEs and provides a column vector of temperatures
%   at the next timestep.

    A = buildMatrix(Nx,Ny);
    dT = A*t;
    T = t + dt.*dT;
end