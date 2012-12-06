function [ B ] = buildSolution( Nx,Ny )
%BUILDSOLUTION Builds a solution column vector
%   Resulting solution vector is composed of values of the function of 
%   second partial derivative at discrete grid points (i*/(Nx+1),j/(Ny+1))
    B=zeros(Nx*Ny,1);
    for i=1:Nx
        for j=1:Ny
            B((j-1)*Nx+i) = -2*pi^2*sin(pi*i/(Nx+1))*sin(pi*j/(Ny+1));
        end
    end
end