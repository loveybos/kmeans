clear all; close all; clc
k=4;               % 图像数据分4类

%% 读取并显示图像(jpg格式，三维)
A=imread('pic_2.jpg','jpg');
B=rgb2gray(A);    % RGB转为灰度图像，二维
imwrite(B,'pic_gray.jpg','jpg');
figure(1);
imagesc(A);
title('图像原图');
%% 图像数据的前期处理(求最大距离，最大值等)
ima=double(B);
copy=ima;         % 灰度图像备份
ima=ima(:);       % 二维图像转为列向量(按列转换)
mi=min(ima);      % 找到图像数据最小值（处理负数用）
ima=ima-mi+1;     % 整体图像值加上最小值绝对值，再加1（正数相当于整体加1）
s=length(ima);    % 图像的长度，如400*400的为160000

m=max(ima)+1;     % m为图像数据的最大值加1，用来计算4个分类阈值（聚类的重心）
h=zeros(1,m);     % 创造一个行向量，有257个值，现在全为0
hc=zeros(1,m);    % 创造一个行向量，有257个值，现在全为0

% 图像的值（从1到257）各有多少个，赋给h行向量。用来计算聚合范围内的均值
for i=1:s
  if ima(i)>0
      h(ima(i))=h(ima(i))+1; 
  end;
end

ind=find(h);      % 找到非负的列，（因为h全为正）这里ind=1,2,3,4,5...257
hl=length(ind);   % ind的长度为hl，这里为257

%% 计算初始聚合重心值，分4类，所以有4个值
mu=(1:k)*m/(k+1); % 最大值除以5，再乘以1,2,3,4得到4个初始聚合重心

%% K-means算法重心值迭代计算过程
while(true)
  oldmu=mu;       % 设置一个对比值

  for i=1:hl      % 从1到257，按聚合重心值把hc分为4类（对整个图像分类的基础）
      c=abs(ind(i)-mu);
      cc=find(c==min(c));
      hc(ind(i))=cc(1);
  end
  
  for i=1:k,      % 新聚合重心值计算过程 
      a=find(hc==i);
      mu(i)=sum(a.*h(a))/sum(h(a)); % 计算出的新的重心值
  end                               % 这里有点类似聚合值周围均值的感觉
  
  if mu==oldmu    % 如果新的重心值收敛了(不变了),则它为最终计算得到的最佳重心值
      break;
  end;
  
end

%% 整个图像数据掩膜(mask)处理(就是将最佳重心值作为阈值，图像掩膜分4类)
s=size(copy);     %备份数据的长宽，如s=[400,400]
mask=zeros(s);    %400*400的零矩阵
for i=1:s(1),
    for j=1:s(2),
        c=abs(copy(i,j)-mu);
        a=find(c==min(c)); %看当前数据距离哪个重心最近，则掩膜分类为它 
        mask(i,j)=a(1);    %图像掩膜分为4类
end
end

mu=mu+mi-1;       % 恢复真实重心？
%% 利用掩膜对整个图像数据进行分类(最后一步了)
for i=1:k
    mu1(i)=uint8(mu(i)); %最佳重心值double数据转为int型 
end;

q=0;
for i=1:s(1)      %整个循环利用掩膜将数据分成了4类，分别为4个最佳重心值
    for j=1:s(2)
        
        while q<=k
            if mask(i,j)==q
                image(i,j)=mu1(q);
            end;
            q=q+1;
        end;
        
        q=0;
    end;
end;

%% 显示并输出成jpg        
figure(2);
imagesc(image);    %不同颜色显示
colorbar;
%colormap('gray'); %分类后图像不同显示方式，可选择灰度显示
title('k-means分类结果');

imwrite(image,'out_class.jpg','jpg');
