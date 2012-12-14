function [ Y ] = expEulerMethod( Nx,Ny,dt,T )
%EXPEULERMETHOD Summary of this function goes here
%   Detailed explanation goes here

A = buildMatrix(Nx,Ny);
dT = A*T;
Y = T + dt.*dT;
end

