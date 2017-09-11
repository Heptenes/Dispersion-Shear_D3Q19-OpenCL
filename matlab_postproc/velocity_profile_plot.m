% 
clear
close all

fName = 'velocity_field_final.txt';

dataRead = dlmread(['../' fName]);

g = 0.00001;
            % x  y  
systemSize = [6 6 126]; % Minus buffer layer

profileDim = 3;
velDim = 1;

vel3D = reshape(dataRead(:,velDim+3),fliplr(systemSize));

dz = 0.5;

w = systemSize(profileDim)-1;
h = w/2;
z = 0:dz:w/2;

% Newtonian
tau = 1.232421875;
nu = (tau-0.5)/3.0;
%nu = 0.03;
u_n = (0.5*g/nu)*((w/2)^2-z.^2);
u_n = [fliplr(u_n) u_n(2:end)];


%0.00000120

% Power law flow profile
nu_0 = 0.0001;
n = 0.5;

C1 = -(-h*(g*h/nu_0)^(1/n))/((1/n) + 1); 
u_pl = (-z.*(g.*z./nu_0).^(1/n))./((1/n) + 1) + C1; 
u_pl = [fliplr(u_pl) u_pl(2:end)];

% Casson
% Casson flow profile
nu_inf = 0.1;
sigma_y = 0.0005;
z_y = sigma_y/g;

C3 = -(1/nu_inf)*(-0.5*g*h^2 + (4/3)*sqrt(g*sigma_y)*(h^(3/2)) - sigma_y*h);
u_c = (1/nu_inf)*(-0.5.*g.*z.^2 + (4/3).*sqrt(g*sigma_y).*(z.^(3/2)) - sigma_y.*z) + C3;

u_c = [fliplr(u_c) u_c(2:end)];


vLB = vel3D(1:(w+1),1,1)';
zr = dz:dz:systemSize(profileDim)-dz;

plot(0.5:1:w+0.5, vLB-0.5*g, 'o');
hold on;
plot(zr,u_n);
%plot(zr,u_pl);
%plot(zr,u_c, '-*');

grid on