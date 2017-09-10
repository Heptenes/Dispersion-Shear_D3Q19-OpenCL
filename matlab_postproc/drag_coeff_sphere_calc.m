
clear

d = 14.0;
rho = 1;

tau = 1.0;
nu = (1/3)*(tau - 0.5); % 3*tau + 1/2

%v = 0.00564692;

re = 2.0;

v = re*nu/d;

dc1 = drag_coeff(re);
dc2 = drag_coeff_2(re);

f = 0.621582;

rad = d/2;
dclb = 2.*f./(rho*v.*v.*pi*rad*rad);

function dc = drag_coeff(re)

dc = (24./re).*(1+0.1806*re.^0.6459) + 0.4251./(1+6880.95./re);

end

function dc = drag_coeff_2(re)

dc = (24./re).*(1+0.27.*re).^0.43 + 0.47.*(1-exp(-0.04.*re.^0.38));

end

