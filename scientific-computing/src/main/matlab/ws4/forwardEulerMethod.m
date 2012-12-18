function [ T ] = forwardEulerMethod( Nx,Ny,dt,t )
%forwardEulerMethod Calculates temperatures at the next timestep
%   Solves a system of ODEs and provides a column vector of temperatures
%   at the next timestep.

    T = zeros(Nx*Ny,1);
    c1 = dt*(Nx + 1)^2; c2 = dt*(Ny + 1)^2; c3 = 1-2*(c1+c2);
    
    for i=1:Nx
        for j=1:Ny
            sum = 0;
            if i > 1 ; sum = sum + c1*t((j-1)*Nx + i-1); end
            if i < Nx; sum = sum + c1*t((j-1)*Nx + i+1); end
            if j > 1 ; sum = sum + c2*t((j-2)*Nx + i); end
            if j < Ny; sum = sum + c2*t((j)*Nx + i); end
            T((j-1)*Nx+i) = sum+c3*t((j-1)*Nx + i);
        end
    end
end