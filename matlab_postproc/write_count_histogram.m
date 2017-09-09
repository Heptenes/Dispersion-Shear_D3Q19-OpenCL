clear

data = dlmread('out.txt');

nodes = zeros(max(data(:,1)),1);

for i = 1:size(data,1)
    if data(i,2) > nodes(i)
        nodes(data(i,1)) = data(i,2);
    end
end

histogram(nodes(nodes>0),1:10);

set(gca,'FontName', 'Clear Sans', 'FontSize',12,'Linewidth',2);