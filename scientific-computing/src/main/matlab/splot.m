function splot(x, f_handler, same_fig)
%splot(x, func, same_fig) Styled print of a function
%   Performs styled plot of the function in a new window depending on 
%   the same_fig flag.

    if ~same_fig 
        figure;
    end
    f = plot (x, f_handler(x));
    
    title(func2str(f_handler));
    set(gcf,'Color',[0.706 0.902 0.961]);
    set(gca,'Color',[0.961 0.953 0.686]);
    set(gca,'XColor',[1 0.2 0.3]);
    set(gca,'YColor',[1 0.2 0.3]);
    set(gca,'ZColor',[1 0.2 0.3]);
    set(f,'Color',[0 0 0]);
    grid on;
end