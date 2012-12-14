function [ E ] = rms( p_approx, p_exact, dt )
%RMS(p_approx, p_exact, dt ) Approximation error calculator
%   Determines RMS error with slight shift towards approximation
    
    p_av = 0;
    for n = 1 : size(p_approx, 2)
        p_av = p_av + (p_approx(n) - p_exact(n)).^2;
    end
    E = sqrt(dt./5.*p_av);
end