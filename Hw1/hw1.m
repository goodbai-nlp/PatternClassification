clear;
close all;

%###################################
% Parzen和k最近邻估计实验
mu = [0,2;1,1.0];
sigma1 = [0.15,0;0,0.15];
sigma2 = [0.45,0.15;0.15,0.25];

% 生成高斯分布随机数 保存到 train.mat
N = 500;
% r1 = mvnrnd(mu(1,:),sigma1,N);
% r2 = mvnrnd(mu(2,:),sigma2,N);
% save train.mat r1 r2;

load train.mat


plot(r1(:,1),r1(:,2),'r+');
xlabel('x轴'),ylabel('y轴');
title('两类训练样本分布');
hold on;
plot(r2(:,1),r2(:,2),'b*');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 获取范围
tmp = sort(r1(:,1));
xl1 = tmp(1); xh1 = tmp(end);
tmp = sort(r1(:,2));
yl1 = tmp(1); yh1 = tmp(end);
tmp = sort(r2(:,1));
xl2 = tmp(1); xh2 = tmp(end);
tmpp = sort(r2(:,2));
yl2 = tmp(1); yh2 = tmp(end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 画出原分布图
xx1 = xl1:0.01:xh1;
yy1 = yl1:0.01:yh1;
xx2 = xl2:0.01:xh2;
yy2 = yl2:0.01:yh2;

figure
fig1 = drawGaussian(mu(1,:),sigma1,xx1,yy1);
set(fig1,'FaceColor','white','EdgeColor','red');

hold on;
fig2 = drawGaussian(mu(2,:),sigma2,xx2,yy2);
set(fig2,'FaceColor','white','EdgeColor','blue'); 
title('原数据分布');
xlabel('x轴'),ylabel('y轴');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%画出估计的分布图
data(:,:,1) = r1;
data(:,:,2) = r2;

%%%%%%%%%%%%%%%%%%%%%以下两部分由于训练样本规模问题,计算量较大,运行时间很长,先注释
% %Parzen窗估计
% drawpic(xx1,yy1,data,1,0.05,0);
% drawpic(xx1,yy1,data,1,0.1,0);
% drawpic(xx1,yy1,data,1,0.125,0);
% 
% %KNN估计
% drawpic(xx1,yy1,data,2,0,3);
% drawpic(xx1,yy1,data,2,0,5);
% drawpic(xx1,yy1,data,2,0,10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%改变协方差矩阵生成test数据 class1,class2 各10个 

figure
testNum = 50;

%%%%%%%%%%%%生成测试数据 保存到 test.mat
% sigma1 = [0.5,0;0,0.5];
% sigma2 = [0.5,0.15;0.15,0.5];
% r1test = mvnrnd(mu(1,:),sigma1,testNum);
% r2test = mvnrnd(mu(2,:),sigma2,testNum);
% 
% save test.mat r1test r2test

load test.mat
alltest = [r1test;r2test];

plot(r1(:,1),r1(:,2),'r+');
xlabel('x轴'),ylabel('y轴');
title('测试样本分布');
hold on;
plot(r2(:,1),r2(:,2),'*');
hold on;
plot(r1test(:,1),r1test(:,2),'go');
hold on;
plot(r2test(:,1),r2test(:,2),'go');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%使用上述两种方法进行分类100

% %%%%%%%%%%%%%%%%%%%%%%%%%%parzen 
figure
xd = [0.01,0.1,0.2,0.3,0.5,0.9,1.0];
[~,sizex] = size(xd); 
res = zeros(1,sizex);
for i=1:sizex
    parzen = xd(i);
    res(i) = testAndEval(data,alltest,1,parzen,0);
end

values = spcrv([[xd(1) xd xd(end)];[res(1) res res(end)]],3);
plot(values(1,:),values(2,:), 'g');
title('parzen窗法估计结果曲线');
xlabel('x: 窗宽'),ylabel('y: 准确率')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%KNN
figure
%xd = [2,3,4,5,6,0.9,1.0];
xd = 2:2:15;
[~,sizex] = size(xd); 
res = zeros(1,sizex);
for i=1:sizex
    k = xd(i);
    res(i) = testAndEval(data,alltest,2,0,k);
end

values = spcrv([[xd(1) xd xd(end)];[res(1) res res(end)]],3);
plot(values(1,:),values(2,:), 'g');
title('KNN法估计结果曲线');
xlabel('x: K值'),ylabel('y: 准确率')


function acc = testAndEval(trainData,testData,method,parzen,k)
%   trainData : 训练数据
%   testData  : 测试数据
%   method    : 估计方法 1:parzen 2:KNN
%   parzen    : parzen窗的窗口
%   k         : knn 的参数k
    [len,~] = size(testData);
    count = 0;
    for i=1:len
        testx = testData(i,:);
        if method==1
            p = Parzen(trainData,testx,parzen,2);
        else
            p = KNN(trainData,k,testx);
        end
        if(p(1)>p(2))
%             plot(testx(1),testx(2),'ro');
            disp(['点',num2str(i),'：[',num2str(testx(1)),num2str(testx(2)),']属于第一类']);
            if i<= len/2
                count=count+1;
            end
        else
%             plot(testx(1),testx(2),'bo');
            disp(['点',num2str(i),'：[',num2str(testx(1)),num2str(testx(2)),']属于第二类']);
            if i> len/2
                count=count+1;
            end
        end
    end
    acc = count/len;
end
