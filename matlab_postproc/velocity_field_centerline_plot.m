% 
clear
close all

fName = 'velocity_field_centerline.txt';

dataRead = dlmread(fName);

dataSize = [65 65];

xSurf = reshape(dataRead(:,4),dataSize);
ySurf = reshape(dataRead(:,5),dataSize);
zSurf = reshape(dataRead(:,6),dataSize);

fig_w = 1024;
fig_h = 360;
figure('Units','points','Position',[0 0 fig_w fig_h],'PaperPositionMode','auto');

subplot(1,2,1)
pcolor(0:(dataSize(1)-1), 0:(dataSize(2)-1), xSurf)
shading interp
colormap(cubehelix([],0.2,-1.4,1.8,1.0,[0,1],[1,0]))
viscircles((dataSize-1)/2, 5, 'Color','w','LineWidth', 1)
colorbar
axis equal
set(gca, 'Color', 'none')

set(gca,'FontName', 'Clear Sans', 'FontSize',16,'Linewidth',1.5);
xlabel(gca,'X (lattice units)', 'Fontsize', 18, 'FontName', 'Clear Sans');
ylabel(gca,'Z (lattice units)', 'Fontsize', 18, 'FontName', 'Clear Sans');
xticks(0:8:64)
yticks(0:8:64)

subplot(1,2,2)
pcolor(0:(dataSize(1)-1), 0:(dataSize(2)-1), zSurf)
shading interp
colormap(cubehelix([],0.2,-1.4,1.8,1.0,[0,1],[1,0]))
%colormap(cubehelix([],0.5,-1.5,1,1,[0.29,1])) 
viscircles((dataSize-1)/2, 5, 'Color','w','LineWidth', 1)
colorbar
axis equal
set(gca, 'Color', 'none')

set(gca,'FontName', 'Clear Sans', 'FontSize',16,'Linewidth',1.5);
xlabel(gca,'X (lattice units)', 'Fontsize', 18, 'FontName', 'Clear Sans');
ylabel(gca,'Z (lattice units)', 'Fontsize', 18, 'FontName', 'Clear Sans');
xticks(0:8:64)
yticks(0:8:64)

