function fig = sfigure(handle, name)
%sfigure Styled figure
%   creates a figure and adds some style to it

    if isempty(handle)
        fig = figure;
    else
        fig = handle;
    end
    figure(fig);
    
    if ~isempty(name)
        set(fig, 'name', name, 'numbertitle','off')
    end
    
    set(gcf,'Color',[0.706 0.902 0.961]);
    set(gca,'Color',[0.961 0.953 0.686]);
    set(gca,'XColor',[1 0.2 0.3]);
    set(gca,'YColor',[1 0.2 0.3]);
    set(gca,'ZColor',[1 0.2 0.3]);
    grid on;
end