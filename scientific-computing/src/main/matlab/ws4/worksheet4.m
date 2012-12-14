% Script 4
% =========================================================================
close all;
clear all;
clc;
%--------------------------------------------------------------------------
% Initial conditions
%--------------------------------------------------------------------------
hEq = @(x,y) (sin(pi.*x).*sin(pi.*y));
hPde = @(x,y) (-2*pi^2*sin(pi*x)*sin(pi*y));

N = [3 7 15 31];

dt = 2.^-(6:12);
t = (1:4)./8;

omitFEM = fase; omitBEM=false;

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
                subplot(length(N), length(dt), (i-1)*length(N)+j);
    %             figure('Name', ['t=',num2str(k),'/8 ','N=',num2str(N(i)),' dt=1/',num2str(2^(j+5))]);
                surf(X,Y,Z); axis([0 1 0 1 0 1]);
            end
        end
    end
end

%--------------------------------------------------------------------------
% Computation using Backward Euler Method
%--------------------------------------------------------------------------
if ~omitBEM
    for k = 1:length(t)
        for i = 1:length(N)
            T=ones(N(i)^2,1);
            ct = t(k);
            while ct > 0
                T = backwardEulerMethod(N(i),N(i),dt(1),T);
                ct = ct-dt(1);
            end    

            ls = linspace(0,1,N(i)+2);
            [X,Y] = meshgrid(ls,ls);
            Z = meshWrapper(T,N(i),N(i));
            figure('Name', ['t=',num2str(k),'/8 ','N=',num2str(N(i)),' dt=1/',num2str(2^6)]);
            surf(X,Y,Z); axis([0 1 0 1 0 1]);
        end
    end
end