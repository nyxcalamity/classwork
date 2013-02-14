function [ A ] = buildMatrix( Nx,Ny )
%buildMatrix Builds up a full matrix of coefficients for heat equation
%   Resulting matrix will have a size of Nx*Ny and will be a full matrix
%   even thouggh most elements might be zero. It will look as follows:
%   
%   Original:       Resultant:
%   A11 A12         A11 A21 A12 A22
%   A21 A22     =>  A11 A21 A12 A22
%                   A11 A21 A12 A22
%                   A11 A21 A12 A22
    c1 = (Nx + 1)^2;
    c2 = (Ny + 1)^2;
    c3 = -2*(c1+c2);
    A = zeros(Nx*Ny);

    for j = 1:Ny
        for i = 1:Nx
            idx = (j-1)*Nx+i;

            if j > 1 ; A(idx, (j-2)*Nx + i) = c2; end
            if j < Ny; A(idx, (j)*Nx + i) = c2; end
            if i > 1 ; A(idx, (j-1)*Nx + i-1) = c1; end
            if i < Nx; A(idx, (j-1)*Nx + i+1) = c1; end
            A(idx, idx) = c3;
        end
    end
end