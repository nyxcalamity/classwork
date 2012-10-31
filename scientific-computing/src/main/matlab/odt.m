function v = ode(f, f0, dt, t_end, method)
% ode(f, y0, dt, t_end, method) Performs analysis of the function
%   Takes in the function, starting value, step end point and the method 
%   which should be used for the calculations and returns a vector of 
%   computed values.
%
%   Posssible methods: euler, heun, runge_kutta

    switch method
        case 'euler'
            v = euler_method(f, f0, dt, 0, t_end);
        case 'heun'
            v = heun_method(f, f0, dt, 0, t_end);
        case 'runge_kutta'
            v = runge_kutta_method(f, f0, dt, 0, t_end);
        otherwise
            disp('Please, input proper method name. Refer to help for more info.');
            return;
    end
end

%--------------------------------------------------------------------------
function v = euler_method(f, f0, dt, t_start, t_end)
    vector_size = (t_end ./ dt);
    v = zeros(1, vector_size);
    v(1) = f0; % we already know the first value
    for n = 2 : vector_size
        v(n) = v(n-1) + dt .* f(v(n-1));
    end
end

%--------------------------------------------------------------------------
% function v = heun_method(f, f0, dt, t_start, t_end)
%     v = [0 0];
% end

%--------------------------------------------------------------------------
% function v = runge_kutta_method(f, y0, dt, t_start, t_end)
%     v = [0 0];
% end