function [ L,U ] = matrixDecomposer( A )
%MATRIXDECOMPOSER Decomposes matrix into lower and upper triangular components
%   Performs decomposition such that outputs L + U = A. Works only with
%   square matrices.
    l = length(A);

    L = zeros(l);
    U = zeros(l);
    
    for i = 1:l
        L(i,1:i) = A(i,1:i);
        U(i,i+1:l) = A(i,i+1:l);
    end
end

