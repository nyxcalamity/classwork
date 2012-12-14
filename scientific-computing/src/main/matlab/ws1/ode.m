function v = ode(fd1, f0, dt, t_end, method)
% ode(fd1, f0, dt, t_end, method) Performs analysis of the function
%   Takes in the function, starting value, step end point and the method 
%   which should be used for the calculations and returns a vector of 
%   computed values.
%
%   Posssible methods: euler, heun, runge_kutta

    switch method
        case 'euler'
            v = eulerMethod(fd1, f0, dt, t_end);
        case 'heun'
            v = heunMethod(fd1, f0, dt, t_end);
        case 'runge_kutta'
            v = rungeKuttaMethod(fd1, f0, dt, t_end);
        otherwise
            disp('Please, input proper method name. Refer to help for more info.');
            return;
    end
end

%--------------------------------------------------------------------------
function v = eulerMethod(fd1, f0, dt, t_end)
    vector_size = (t_end ./ dt) + 1;
    v = zeros(1, vector_size);
    v(1) = f0; % we already know the initial value
    for n = 2 : vector_size
        v(n) = v(n-1) + dt .* fd1(v(n-1));
    end
end

%--------------------------------------------------------------------------
% Basically we introduce a mean value of function change frequencies
function v = heunMethod(fd1, f0, dt, t_end)
    vector_size = (t_end ./ dt) + 1;
    v = zeros(1, vector_size);
    v(1) = f0; % we already know the initial value
    for n = 2 : vector_size
        tmp_int_point = v(n-1) + dt .* fd1(v(n-1));
        v(n) = v(n-1) + dt ./ 2 .* (fd1(v(n-1)) + fd1(tmp_int_point));
    end
end

%--------------------------------------------------------------------------
function v = rungeKuttaMethod(fd1, f0, dt, t_end)
    vector_size = (t_end ./ dt) + 1;
    v = zeros(1, vector_size);
    v(1) = f0; % we already know the initial value
    for n = 2 : vector_size        
        k1 = dt.*fd1(v(n-1));
        k2 = dt.*fd1(v(n-1)+k1./2);
        k3 = dt.*fd1(v(n-1)+k2./2);
        k4 = dt.*fd1(v(n-1)+k3);
        
        v(n) = v(n-1) + (k1 + k2.*2 + k3.*2 + k4)./ 6;
    end
end