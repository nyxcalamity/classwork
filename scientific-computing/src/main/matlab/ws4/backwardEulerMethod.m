function [ T ] = backwardEulerMethod( Nx,Ny,dt,t )
%backwardEulerMethod Calculates temperatures at the next timestep
%   Solves a system of ODEs and provides a column vector of temperatures
%   at the next timestep. Uses iterative Gauss-Siedel method to solve the
%   system. Accuracy of calculations is 1e-4, that is residual norm is
%   expected to be below that value for results to be considered as
%   solution.
    
    T = ones(Nx*Ny,1); %guessed value   
    c1 = -(Nx + 1)^2; c2 = -(Ny + 1)^2; c3 = 1/dt-2*(c1+c2);
    
    while true % do-while syntax
        % Iterative calculation of unknown
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if i > 1 ; sum = sum + c1*T((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*T((j-1)*Nx + i+1); end
                if j > 1 ; sum = sum + c2*T((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*T((j)*Nx + i); end
                T((j-1)*Nx+i) = (t((j-1)*Nx+i)/dt - sum)/c3;
            end
        end
        
        % Calculating residual norm
        residualNorm = 0;
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if i > 1 ; sum = sum + c1*T((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*T((j-1)*Nx + i+1); end
                if j > 1 ; sum = sum + c2*T((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*T((j)*Nx + i); end
                residualNorm = residualNorm + (t((j-1)*Nx+i)/dt - sum - c3*T((j-1)*Nx+i))^2;
            end
        end
        residualNorm  = sqrt(residualNorm/(Nx*Ny));
        
        % Exit condition
        if abs(residualNorm) < 1e-4; break; end
    end

end