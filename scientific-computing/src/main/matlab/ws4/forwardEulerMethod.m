function [ T ] = forwardEulerMethod( Nx,Ny,dt,t )
%EXPEULERMETHOD Summary of this function goes here
%   Detailed explanation goes here

    A = buildMatrix(Nx,Ny);
    dT = A*t;
    T = t + dt.*dT;
end

