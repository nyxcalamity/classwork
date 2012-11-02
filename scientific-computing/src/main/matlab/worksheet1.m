% Script
% Calculates approximations of ode solutions and their errors. 
% P.S.: Luckily it's not tested for performance. :)
% =========================================================================
%   
f = @(t) ( (10)./(1+9*exp(-t)) );   % function definition
fd1 = @(p) ( (1 - p./10) .* p );    % first order derivative definition
f0 = 1;                             % initial value
dt = 0.1;                           % step size
t_end = 5;                          % end time

%--------------------------------------------------------------------------
% Plotting the function p(t)
%--------------------------------------------------------------------------
t = 0 : dt : t_end;
fig_main = figure;
plot(t, f(t), 'k');
fig_main = sfigure(fig_main, 'Analytical solution');
legend ('exact');

%--------------------------------------------------------------------------
% Calculating approximate values and plotting respective graphs
%--------------------------------------------------------------------------
dt = [1  1/2 1/4 1/8];
E = zeros(3, 4);
E_approx = zeros(3, 4);

fig_euler = figure;
plot(t, f(t), 'k', 'linewidth', 1.7);
hold on;

fig_heun = figure;
plot(t, f(t), 'k', 'linewidth', 1.7);
hold on;

fig_rk4 = figure;
plot(t, f(t), 'k', 'linewidth', 1.7);
hold on;

color = ['b' 'r' 'g' 'm'];
for n = 1 : size(dt, 2)
    t = 0:dt(n):t_end;
    f_exact = f(t);
    k = dt(n)./dt(size(dt,2)); % number of steps vs smallest step size
    
    % Euler's method and errors -------------------------------------------
    figure(fig_euler);
    f_euler = ode(fd1, f0, dt(n), t_end, 'euler');
    plot(t, f_euler, color(n), 'linewidth', 1.7);
    
    f_euler_best = ode(fd1, f0, dt(size(dt,2)), t_end, 'euler');
    f_reference = zeros(1, size(f_euler, 2));
    for m = 1: size(f_euler,2)        
        f_reference(1, m) = f_euler_best(1, ((m-1).*k+1) );
    end
    
    E(1, n) = rms(f_euler, f_exact, dt(n));
    E_approx(1, n) = rms(f_euler, f_reference, dt(n));
    
    % Heun's method and errors --------------------------------------------
    figure(fig_heun);
    f_heun  = ode(fd1, f0, dt(n), t_end, 'heun');
    plot(t, f_heun, color(n), 'linewidth', 1.7);
    
    f_heun_best  = ode(fd1, f0, dt(size(dt,2)), t_end, 'heun');
    f_reference = zeros(1, size(f_heun, 2));
    for m = 1: size(f_heun,2)        
        f_reference(1, m) = f_heun_best(1, ((m-1).*k+1) );
    end
    
    E(2, n) = rms(f_heun, f_exact, dt(n));
    E_approx(2,n) = rms(f_heun, f_reference, dt(n));   
        
    % Runge-Kutta's method and errors -------------------------------------
    figure(fig_rk4);
    f_rk4 = ode(fd1, f0, dt(n), t_end, 'runge_kutta');  
    plot(t, f_rk4, color(n), 'linewidth', 1.7);
    
    f_rk4_best = ode(fd1, f0, dt(size(dt,2)), t_end, 'runge_kutta');  
    f_reference = zeros(1, size(f_rk4, 2));
    for m = 1: size(f_rk4,2)        
        f_reference(1, m) = f_rk4_best(1, ((m-1).*k+1) );
    end
    
    E(3, n) = rms(f_rk4, f_exact, dt(n));
    E_approx(3, n) = rms(f_rk4, f_reference, dt(n));
end

fig_euler = sfigure(fig_euler, 'Eulers method');
legend ('exact', '1', '1/2', '1/4', '1/8');
fig_heun = sfigure(fig_heun, 'Heuns method');
legend ('exact', '1', '1/2', '1/4', '1/8');
fig_rk4 = sfigure(fig_rk4, 'Runge-Kuttas method');
legend ('exact', '1', '1/2', '1/4', '1/8');

%--------------------------------------------------------------------------
% Printing errors
%--------------------------------------------------------------------------
disp('Eulers method: errors of approximation (desc step):');
disp (E(1,:));
disp('Heuns method: errors of approximation (desc step):');
disp (E(2,:));
disp('Runge-Kuttas method: errors of approximation (desc step):');
disp (E(3,:));

disp('Eulers method: error decrease factor (desc step):');
disp(E(1, 1:3) ./ E(1, 2:4));
disp('Heuns method: error decrease factor (desc step):');
disp((E(2, 1:3) ./ E(2, 2:4)));
disp('Runge-Kuttas method: error decrease factor (desc step):');
disp((E(3, 1:3) ./ E(3, 2:4)));

disp('Eulers method: approximate errors of approximation (desc step):');
disp (E_approx(1,:));
disp('Heuns method: approximate errors of approximation (desc step):');
disp (E_approx(2,:));
disp('Runge-Kuttas method: approximate errors of approximation (desc step):');
disp (E_approx(3,:));