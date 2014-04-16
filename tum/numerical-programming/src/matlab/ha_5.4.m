% Home Assignment 5.4
% =========================================================================
f = @(x)(1./(1+x.^2));
a = -5; b = 5;
n = [5 10 20 40];

x_exact = -5:0.01:5;
y_exact = f(a);
%--------------------------------------------------------------------------
% Estimation of interpolation error using three different approaches
%--------------------------------------------------------------------------
e = zeros(1,3);
for i=1:length(n)
	%Equally spaced
	x1 = linspace(a,b,n(i));
	y1 = f(x1);
	p1 = polyfit(x1,y1,n(i));
	e(1) = max(abs(polyval(p1,x_exact) - y_exact));

	%Chebyshev nodes
	x2 = chebyshev_nodes(a,b,n(i));
	y2 = f(x2);
	p2 = polyfit(x2,y2,n(i));
        e(2) = max(abs(polyval(p2,x_exact) - y_exact));

	%Cubic spline
	s = spline(x1,y1);
	d = ppval(s,x_exact);
	e(3) = max(abs(d - y_exact));

	disp('Computation of interpolation error for n:');
	disp(n(i));
	disp('');
	disp(e);
end
