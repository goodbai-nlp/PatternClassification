function [ p ] = Parzen( w,x,h,f )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%   w: 所有训练样本 分为a,b,c 三类
%   x: 测试样本
%   h: 窗口大小
%   f: 窗口函数 f=1:矩形窗 f=2: 高斯窗
%   p: p(i)每个类别的概率
[xx,~,zz] = size(w);

p = zeros(1,zz);

for i= 1:zz     % 类别
    hn = h;
    
    for j = 1:xx
%         hn = hn/sqrt(j);
        if f == 2   %高斯窗
             p(i) = p(i) + exp(-(x - w(j,:,i))*(x - w(j,:,i))'/ (2 * power(hn,2))) / (hn * sqrt(2*pi));
        end
    end
    p(i) = p(i)/xx;
end
end


