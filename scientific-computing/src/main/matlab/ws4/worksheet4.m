% Script 4
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

omitFEM=false; omitBEM=true;

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
        print('-dpng',strcat('t=',num2str(k),':8'));
    end
end

%--------------------------------------------------------------------------
% Computation using Backward Euler Method
%--------------------------------------------------------------------------
if ~omitBEM
    for k = 1:length(t)
        figure('Name', ['t=',num2str(k),'/8 ',' dt=1/',num2str(2^6)]);
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
            subplot(1,length(N),i);
            surf(X,Y,Z); %axis([0 1 0 1 0 1]);
        end
    end
end