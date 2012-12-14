function [ T ] = backwardEulerMethod( Nx,Ny,dt,t )
%BACKWARDEULERMETHOD Summary of this function goes here
%   Detailed explanation goes here
    
    T = zeros(Nx*Ny,1); %guessed value   
    c1 = (Nx + 1)^2; c2 = (Ny + 1)^2; c3 = -2*(c1+c2);
    
    while true
        % Iterative calculation of unknown
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if j > 1 ; sum = sum + c2*T((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*T((j)*Nx + i); end
                if i > 1 ; sum = sum + c1*T((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*T((j-1)*Nx + i+1); end
                T((j-1)*Nx+i) = (t((j-1)*Nx+i) + dt*sum)/(1-dt*c3);
            end
        end
        
        % Calculating residual norm
        residualNorm = 0;
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if j > 1 ; sum = sum + c2*T((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*T((j)*Nx + i); end
                if i > 1 ; sum = sum + c1*T((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*T((j-1)*Nx + i+1); end
                residualNorm = residualNorm + (t((j-1)*Nx+i) + dt*sum - (1-dt*c3)*T((j-1)*Nx+i))^2;
            end
        end
        residualNorm  = sqrt(residualNorm/(Nx*Ny));
        
        % Exit condition
        if abs(residualNorm) < 0.0001; break; end
    end

end

