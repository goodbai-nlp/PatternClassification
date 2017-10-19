function [ p ] = KNN( w,k,x )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%   w: 所有训练样本 分为a,b,c 三类
%   x: 测试样本
%   k: k近邻
%   p: p(i)每个类别的概率

[xx,yy,zz] = size(w);
p = zeros(1,zz);
for i= 1:zz     % 类别
    num = k;
    dist = zeros(1,xx);
    for j = 1:xx
        dist(j) = norm(x - w(j,:,i),yy);
    end
    t = sort(dist);
    maxd = t(num);
    vn = sqrt(power(3.141592,yy))*power(maxd,yy)/gamma(yy/2+1);
    p(i) = k/(xx * vn);
end
end
