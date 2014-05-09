% Worksheet 1
% NOTE: It was performed on Linux, on other platforms problems with excel 
% parsing might arise.
% =========================================================================
close all;
clear all;
clc;
%--------------------------------------------------------------------------
% Assignment 1.0
%--------------------------------------------------------------------------
A=[0 2 -1; 2 -1 1; 2 -1 3];
disp(A);
B=[1 2 4; 11 8 3; 9 5 2];
disp(B);
disp('Entry in the second row and first column of A:');
disp(A(2,1));
disp('Vector consisting of the first row of A:');
disp(A(1,:));
disp('Vector consisting of the third column of A:');
disp(A(:,3));
disp('The row sums of A:');
disp(sum(A));
disp('The mean values of the columns of A:');
disp(mean(A));
disp('A vector consisting of all entries of A which are smaller than 2:');
disp(A(A<2));
disp('The transpose of A:');
disp(A');
disp('The matrix multiplication of A and B:');
disp(A*B);
disp('A matrix consisting of the elementwise multiplication of A and B:');
disp(A.*B);

disp('a with column wise median of B:');
a=median(B);
disp(a);
%--------------------------------------------------------------------------
% Assignment 1.1 Characteristics of stock returns
%--------------------------------------------------------------------------
[num,text,raw] = xlsread('data/dow-jones.xlsx');
meanValue  = mean(num(:,2));
logReturns = [0; diff(num(:,2))];
meanReturn = mean(logReturns);
entries    = length(logReturns);
variance   = sum((logReturns-meanReturn).^2)/entries;
std        = sqrt(variance);
skewness   = sum(((logReturns-meanReturn)./std).^3)/entries;
kurtosis   = (sum((logReturns-meanReturn).^4)/entries) / (variance^2); %leptokurtosis in our case?

disp('Parameters of the DJ historic data: ');
fprintf('Mean: %f Std: %f Variance: %f Skewness: %f Kurtosis: %f\n', meanReturn, std, variance, skewness, kurtosis);
fprintf('Mean value: %f Std(percent): %f\n', meanValue, (std*100/meanValue));

histfit(logReturns);
xlim([-0.05, 0.05]);
%TODO: d and c tasks, check that data makes sense
%plot3(num(:,1), num(:,2), logReturns)
%plot(logReturns)
%--------------------------------------------------------------------------
% Assignment 1.2 Portfolio optimization | Hint:quadprog
%--------------------------------------------------------------------------
numStocks=4;
muTotal = 0.17; %total expected return of portfolio
mu = [0.05 -0.2   0.15  0.3]; %expected return per stock
C  = [0.08 -0.05 -0.05 -0.05; ... %covariance of returns of stock i and j
     -0.05  0.16 -0.02 -0.02; ...
     -0.05 -0.02  0.35  0.06; ...
     -0.05 -0.02  0.06  0.35];

f = zeros(1,numStocks);
A = -mu; b = -muTotal; %A*x<=b => -A*x>=-b
Aeq = ones(1,numStocks); beq = 1; %Aeq*x=beq
lb = zeros(1,numStocks); ub = ones(1,numStocks); %lower and upper boundaries

tic
x = quadprog(C,f,A,b,Aeq,beq,lb,ub);
elapsedTime = toc;

disp('Optimal stock weights are: ');
disp(x);
fprintf('Computation took (s): %d\n', elapsedTime);