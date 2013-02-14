% Script 4 | Group 9 
% Solving instationary equations using Euler's methods.
% =========================================================================
close all;
clear all;
clc;
%--------------------------------------------------------------------------
% Initial conditions
%--------------------------------------------------------------------------
N = [3 7 15 31];

dt = 2.^-(6:12);
t = (1:4)./8;

omitFEM=false; omitBEM=false;

%--------------------------------------------------------------------------
% Computation using Forward Euler Method
%--------------------------------------------------------------------------
if ~omitFEM
    for k = 1:length(t)
        figure('Name', ['t=',num2str(k),'/8 ']);
        for i = 1:length(N)
            for j=1:length(dt)
                T=ones(N(i)^2,1);
                ct = t(k);
                while ct > 0
                    T = forwardEulerMethod(N(i),N(i),dt(j),T);
                    ct = ct-dt(j);
                end    

                ls = linspace(0,1,N(i)+2);
                [X,Y] = meshgrid(ls,ls);
                Z = meshWrapper(T,N(i),N(i));
                subplot(length(N), length(dt), (i-1)*length(dt)+j);
                surf(X,Y,Z); % axis([0 1 0 1 0 1]);
            end
        end
        print('-dpng',strcat('figure_t',num2str(k)));
    end
end

%--------------------------------------------------------------------------
% Computation using Backward Euler Method
%--------------------------------------------------------------------------
if ~omitBEM
    figure('Name', ['Implicit Eulers Method for dt=1/',num2str(2^6)]);
    idx_dt=1;
    for k = 1:length(t)
        for i = 1:length(N)
            T=ones(N(i)^2,1);
            ct = t(k);
            while ct > 0
                T = backwardEulerMethod(N(i),N(i),dt(idx_dt),T);
                ct = ct-dt(idx_dt);
            end    

            ls = linspace(0,1,N(i)+2);
            [X,Y] = meshgrid(ls,ls);
            Z = meshWrapper(T,N(i),N(i));
            subplot(length(t),length(N),(k-1)*length(t)+i);
            surf(X,Y,Z); %axis([0 1 0 1 0 1]);
            title(['t=',num2str(t(k)),' N=',num2str(N(i))]);
        end
    end
end