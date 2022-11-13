

 % add Daniela-----------------------------------

clear all
clc

load class
load data
k=5

range=1:100;
[takeout0]=takeoutData(data,class,range,k); hold on
range=101:200;
[takeout1]=takeoutData(data,class,range,k); hold on
range=201:300;
[takeout2]=takeoutData(data,class,range,k); hold on
range=301:400;
[takeout3]=takeoutData(data,class,range,k); hold on
range=401:500;
[takeout4]=takeoutData(data,class,range,k); hold on
range=501:600;
[takeout5]=takeoutData(data,class,range,k); hold on
range=601:700;
[takeout6]=takeoutData(data,class,range,k); hold on
range=701:800;
[takeout7]=takeoutData(data,class,range,k); hold on
range=801:900;
[takeout8]=takeoutData(data,class,range,k); hold on
range=901:1000;
[takeout9]=takeoutData(data,class,range,k); hold on
% sample=101:200;
% [scatteR]=plotscatter(sample,data,class)
%  

function [takeout]=takeoutData(data,class,range,k)
number=(range(1,1)-1);
TClass=unique(class);
LaC=length(TClass);
 I=zeros(k,LaC);
 P=zeros(k,LaC);
[euc_class]=Distance(data(:,1:2,range),data(:,1:2,range));
     [P,I] = maxk(euc_class(:,:),k);
     M=unique(I)';
     m=length(M);
     Count=zeros(1,m);
     for i=1:m
         Count(:,i)=sum(sum(I(:,:)==M(:,i)));
     end
     MCount=[M;Count]';
     MCount = sortrows(MCount,2,'descend'); 
     takeout=MCount(1:k,1);
     takeout=takeout+number;
     sample=takeout'
[scatteR]=plotscatter(sample,data,class)
end
function [euc]=Distance(traindata,data) %----Daniela----------------------
euc=zeros(100,100);
for i=1:100
    for j=1:100
       euc(i,j)=sum(diag(pdist2(traindata(:,:,i),data(:,:,j))));
    end
end
end  %-----------------------------------------------------
function [scatteR]=plotscatter(sample,data,class)
n=length(sample);
for i=1:n
    e=sample(:,i);
    nexttile
    scatteR=scatter(data(:,1,e),data(:,2,e))
    c=class(:,e);
    title(['Class ',num2str(c)])
end
end