function x = chebyshev_nodes(a, b, n)
% chebyshev_nodes(a, b, n) Computes n Chebyshev nodes on an interval [a,b].
	s1 = (a+b)/2;
	s2 = (b-a)/2;
        i = 1:n;
	x = s1+s2*cos( (2.*i - 1)/(2*n) .* pi );
end

