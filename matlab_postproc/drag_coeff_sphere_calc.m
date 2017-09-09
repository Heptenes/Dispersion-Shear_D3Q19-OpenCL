
clear

d = 10.0;
v = 0.01;
eta = 1/6;

re = d*v/eta;

drag_coeff(re)


%f = 0.178614; % 10000
%f = 0.179576; % 20000
f = 0.183;

rho = 1;

dclb = 8*f/(1*v*v*pi*d*d);

function dc = drag_coeff(re)

dc = (24/re)*(1+0.1806*re^0.6459) + 0.4251/(1+6880.95/re);

end
