function v = ode(fn, y0, dt, t_end, method)
% ode(fn, y0, dt, t_end, method) Performs analysis of the function
%   Takes in the function, starting value, step end point and the method 
%   which should be used for the calculations and returns a vector of 
%   computed values.
%
%   Posssible methods: euler, heun, runge_kutta

    switch method
        case 'euler'
            v = euler_method(fn, y0, dt, t_end);
        case 'heun'
            v = heun_method(fn, y0, dt, t_end);
        case 'runge_kutta'
            v = runge_kutta_method(fn, y0, dt, t_end);
        otherwise
            disp('Please, input proper method name. Refer to help for more info.');
            return;
    end
end

%--------------------------------------------------------------------------
function v = euler_method(fn, y0, dt, t_end)
    v = [0 0];
end

%--------------------------------------------------------------------------
function v = heun_method(fn, y0, dt, t_end)
    v = [0 0];
end

%--------------------------------------------------------------------------
function v = runge_kutta_method(fn, y0, dt, t_end)
    v = [0 0];
end