% Worksheet 1 
% =========================================================================
close all;
clear all;
clc;
%--------------------------------------------------------------------------
% Initial conditions
%--------------------------------------------------------------------------
A=[0 2 -1; 2 -1 1; 2 -1 3]
B=[1 2 4; 11 8 3; 9 5 2]
%--------------------------------------------------------------------------
% Computation
%--------------------------------------------------------------------------
disp("Assignment 1.0");
disp("Entry in the second row and first column of A:");
A(2,1)
disp("Vector consisting of the first row of A:");
A(1,:)
disp("Vector consisting of the third column of A:");
A(:,3)
disp("The row sums of A:");
sum(A')'
disp("The mean values of the columns of A:");
mean(A)
disp("A vector consisting of all entries of A which are smaller than 2:");
A(A<2)
disp("The transpose of A:");
A'
disp("The matrix multiplication of A and B:");
A*B
disp("A matrix consisting of the elementwise multiplication of A and B:");
A.*B

disp("a with column wise median of B:")
a=median(B)
%--------------------------------------------------------------------------
disp("Assignment 1.1");

