function x = newton(f, df, x0, n)
% newton(f, df, x0, n) Newton's method approximation of roots of equation.
%	Performs n iterations of Newton's method for finding roots of the provided equation.
	x = zeros(1,n+1);
	x(1) = x0;
	
	for i=1:n
		x(i+1) = x(i) - f(x(i))/df(x(i));
	end
end
