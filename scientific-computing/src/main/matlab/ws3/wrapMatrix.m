function [ W ] = wrapMatrix( X )
%WRAPMATRIX Takes provided matrix and puts zeroes around it's borders.
%   Thus, with for a 2x2 matrix [A11 A12;A21 A22] one will get a 4x4 matrix
%   of the form [0 0 0 0;0 A11 A12 0;0 A21 A22 0;0 0 0 0];
    s = size(X);
    W = zeros(s(1,1)+2,s(1,2)+2);
    
    for i=1:s(1,1)
        for j=1:s(1,2)
            W(i+1,j+1) = X(i,j);
        end
    end
end