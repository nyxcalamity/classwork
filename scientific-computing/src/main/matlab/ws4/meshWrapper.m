function [ W ] = meshWrapper( X,Nx,Ny )
%MESHWRAPPER Maps a vector to a mesh and puts zeroes around it's borders.
%   Thus, for a 4x1 vector [A11 A21 A12 A22] one will get a 4x4 matrix
%   of the form [0 0 0 0;0 A11 A12 0;0 A21 A22 0;0 0 0 0];
    W = zeros(Nx+2,Ny+2);
    
    for i=1:Nx
        for j=1:Ny
            W(i+1,j+1) = X((j-1)*Nx+i);
        end
    end
end