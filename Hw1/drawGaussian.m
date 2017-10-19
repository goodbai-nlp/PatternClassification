function  f  = drawGaussian(u,v,x,y)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%   u: vector,expactation;
%   v: covariance matrix;
%   x: range of x
%   y: range of y

[X,Y]=meshgrid(x,y);
VX=v(1,1);
dx=sqrt(VX);    % x的方差
VY=v(2,2);
dy=sqrt(VY);    % y的方差
Cov = v(1,2);
r=Cov/(dx*dy);
tmp=1/(2*pi*dx*dy*sqrt(1-r^2));
p1=-1/(2*(1-r^2));
px=(X-u(1)).^2./VX;
py=(Y-u(2)).^2./VY;
pxy=2*r.*(X-u(1)).*(Y-u(2))./(dx*dy);
Z=tmp*exp(p1*(px-pxy+py));
% figure
f = mesh(x,y,Z);
end

