% Script 3
% Calculating solution for heat equation and its stationary pde of second 
% prder while storing matrix in several ways and computing errors.
% =========================================================================

hEq = @(x,y) (sin(pi.*x).*sin(pi.*y));
hPde = @(x,y) (-2*pi^2*sin(pi*x)*sin(pi*y));

N = [7 15 31 63];

titles = {'Nx=Ny='; 'Runtime'; 'Matrix Storage'};
computationResults = zeros(6,length(N));

%--------------------------------------------------------------------------
% Calculating matrix
%--------------------------------------------------------------------------
for i=1:length(N)
    Nx = N(i); Ny = N(i);

    A = buildMatrix(Nx,Ny);
    B = buildSolution(Nx,Ny);
    
    % Using full matrix and inbuilt solver
    tic; X = A\B; timeElapsed = toc;
    computationResults(1,i) = timeElapsed;
    computationResults(2,i) = numel(A)+numel(X)+numel(B);
    
    % Using sparse matrix and inbuilt solver
    a = sparse(A);
    tic; X = a\B; timeElapsed = toc;
    computationResults(3,i) = timeElapsed;
    computationResults(4,i) = nnz(a)+numel(X)+numel(B);
    
    % Using knowledge about matrix and Gauss-Siedel method
    tic; X = gaussSiedelSolver(B,Nx,Ny); timeElapsed = toc;
    computationResults(5,i) = timeElapsed;
    computationResults(6,i) = numel(X)+numel(B);
    
    if (i == length(N))
        x = linspace(0,1,Nx+2);
        y = linspace(0,1,Ny+2);
        z = meshWrapper(X,Nx,Ny);
        
        figure; mesh(x,y,z); axis([0 1 0 1 0 1]);
        figure; surf(x,y,z); axis([0 1 0 1 0 1]); view(2);
    end
end

format shortG
disp(computationResults);