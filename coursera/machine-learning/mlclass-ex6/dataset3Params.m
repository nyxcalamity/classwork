function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

if C ~= 1 || sigma ~= 0.1
    eval_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    n = size(eval_params,2);
    err = zeros(n,n);

    for i=1:n
        for j=1:n
            model = svmTrain(X, y, eval_params(i), @(x1, x2) gaussianKernel(x1, x2, eval_params(j)));
            predictions = svmPredict(model, Xval);
            err(i,j) = mean(double(predictions ~= yval));
        end
    end

    [value, idx] = min(err(:));
    i = mod(idx,n);
    j = fix(idx/n)+1;

    if i == 0
         i = n;
    end
    
    C = eval_params(i)
    sigma = eval_params(j)
end
% =========================================================================

end
