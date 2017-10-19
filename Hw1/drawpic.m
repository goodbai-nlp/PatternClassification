function drawpic( xx,yy,data,method,parzen,k)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
[X,Y]=meshgrid(xx,yy);
[l,h] = size(X);

Z = zeros(l,h);

for i=1:l
    for j=1:h
        test = [X(i,j),Y(i,j)];
        if method==1
            z = Parzen(data,test,parzen,2);
        end
        if method==2
            z = KNN(data,k,test);
        end
        Z(i,j) = z(1);
    end
end

figure
mesh(xx,yy,Z);
[xt,~,~] = size(data);
if method==1
    figtitle = [' h = ' num2str(parzen) ',  N=' num2str(xt)];
    % figtitle = [' h = ' num2str(1) ',  N=' num2str(300)];
end
if method==2
    figtitle = [' k = ' num2str(k) ',  N=' num2str(xt)];
end
title(figtitle);
xlabel('x轴'),ylabel('y轴'),zlabel('z轴');

end

