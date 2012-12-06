function [ X ] = gaussSiedelSolver( B,Nx,Ny )
%GAUSSSIEDEL Iteratively finds solution to the system of linear equations.
%   The exit condition or accuracy of the method is set to 10^(-4), 
%   that is if solution converges.
    
    X = zeros(size(B));
    c1 = (Nx + 1)^2;
    c2 = (Ny + 1)^2;
    c3 = -2*(c1+c2);
    
    while true
        % Iterative calculation of unknown
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if j > 1 ; sum = sum + c2*X((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*X((j)*Nx + i); end
                if i > 1 ; sum = sum + c1*X((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*X((j-1)*Nx + i+1); end
                X((j-1)*Nx+i) = (B((j-1)*Nx+i) - sum)/c3;
            end
        end
        
        % Calculating residual norm
        residualNorm = 0;
        for i=1:Nx
            for j=1:Ny
                sum = 0;
                if j > 1 ; sum = sum + c2*X((j-2)*Nx + i); end
                if j < Ny; sum = sum + c2*X((j)*Nx + i); end
                if i > 1 ; sum = sum + c1*X((j-1)*Nx + i-1); end
                if i < Nx; sum = sum + c1*X((j-1)*Nx + i+1); end
                residualNorm = residualNorm + (B(i) - sum - c3*X((j-1)*Nx+i))^2;
            end
        end
        residualNorm  = sqrt(residualNorm/length(B));
        
        % Exit condition
        if abs(residualNorm) < 0.0001; break; end
    end
    
    % Wrap the matrix with 0s
    X = wrapMatrix(X);
end