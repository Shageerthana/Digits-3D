clear all

dimensions = 2;     % dimensions
Rows = 80;          % number of points
t = 0.8;            % trainset dimension
k = 10;             %number of outliers remove
K = 11;             % k NN
          

if ~(exist('data','var')) && ~(exist('class','var'))
    [data,class] = get_all_dataset(Rows,dimensions);
end

% Removing the samples that are outliers-----------------------------------

n = unique(class); %counting the classes that we have
L = length(n); 
takeout = []; %creating a matrix for the samples that we are going to remove
for i = 1:L
    rl = n(i)*100+1; %initial range example 1
    rh = 100*i; %final range example 100
    range = rl:rh; %range --> example 1:100
    [takeout1] = takeoutData(data,class,range,k);  % equation that evaluates within the same class the samples that are less similar to the others
    takeout = [takeout takeout1]; %creating a list of the index of the samples that we are going to take out
end
data(:,:,takeout) = []; %taking the samples out of the data
class(takeout) = []; %taking the class that correspond to samples remove

% Randomize the data-------------------------------------------------------
rand_ind = randperm(size(class,2)); 
data = data(:,:,rand_ind); %randomize the samples in the data
class = class(rand_ind); %randomize accoring to the data

% Dividing train and test--------------------------------------------------

NUM = fix(size(data,3));
traindata = data(:,:,1:NUM*t);
trainclass = class(:,1:NUM*t);
testdata = data(:,:,NUM*t+1:end);
testclass = class(:,NUM*t+1:end);
%--------------------------------------------------------------------------
 

% Taking the zeros out of the knn clasification---------------------------- 
Stp = length(testdata);
tclassN0 = find(trainclass == 0);
traindata1 = traindata;
trainclass1 = trainclass;
traindata1(:,:,tclassN0) = [];
trainclass1(tclassN0) = [];

% Decision tree: if it fits the criteria of zero then it get classify as 
%  zero if not then in get classify by knn method--------------------------
C = zeros(1,Stp); % making a matrix to allocate the classes

for i=1:Stp
    [Cz] = checkif(1,Rows,testdata(:,:,i)); %class for the zero
    if Cz == 0 % if is zero then leave it like cero, if not use the knn
        C(i) = 0;
    else
    C(i) = knn(trainclass1, traindata1, testdata(:,:,i), K);
    end
end

% Checking the results with percentage and confusion matrix----------------
result = evaluate_difference(C, testclass);
confmat = confusionmat(testclass,C);
str = '';
str = sprintf("\nClassification data:\nDimensions used: %d, Number of points: %d,\nRemoved outliers per digit: %d,\nTrainset:Testset = %d:%d,\nk: %d,\nAccuracy: %4.2f%%\n", dimensions, Rows, k, t*100, 100 - t*100, K, result);
figure
confusionchart(confmat, 0:9);
title(str);

%removing temporal variables-----------------------------------------------
clear Cz dimensions i j k L n NUM rand_ind range rh rl Rows Stp t takeout takeout1 tclassN0 testdata tmp tmp0 trainclass1 traindata1

function [data, class] = get_all_dataset(num_points, dim)
% this function returns all the dataset with already preprocessed data,
% while class refers to the class of each sample.
% data will be a pointsxdimx1000 matrix.
% class will be a 1 vector containing the classes of the samples
    data = zeros(num_points, dim, 1000);
    class = zeros(1,1000);
    % data preprocessing
    for i = 0:9
        class(((i*100)+1):(i+1)*100) = (i)*ones(1,100);
        for j = 1:100
            tmp = normalize(get_digits_3D_data(i, j, dim));
            r = size(tmp, 1);
            data(:,:,(i*100)+j) = resample(tmp, num_points, r, dim);
        end
    end
end%-----------------------------------------------------------------------

function C = knn(trainclass, traindata, data, k)
    dataNUM = size(data, 3);
    trainNUM = size(traindata, 3);
    distance_matrix = zeros(dataNUM, trainNUM);
    % evaluating distance_matrix(j,i): having in cell j,i the distance of
    % sample i from traindata j. Distance is defined as sum of the point by
    % point distances
    for i = 1:dataNUM
        for j = 1:trainNUM
            distance_matrix(j,i) = sample_dist(data(:,:,i), traindata(:,:,j));
        end
    end
    
    class = repmat(trainclass, [size(trainclass),1])';
    [~, index] = sort(distance_matrix);
    train = class(index);
    % k_nearest = n_nearest(1:k,:);
    train = train(1:k,:);
    for i = 1:size(data,3)
        for j = 1:max(trainclass)
            tmp(j,i) = sum(train(:,i)==j);
        end
    end
    [~, C] = max(tmp);
end%-----------------------------------------------------------------------

function dist = sample_dist(S1, S2)
% this function returns the sum of the point by point distances between two
% samples. S1 ansd S2 are numpointsxdim matrices
    dist = 0;
    for i = 1:size(S1,1)
        dist = dist + euc_distance(S1(1,:), S2(2,:));
    end
end%-----------------------------------------------------------------------

function dist = euc_distance(P1, P2)
% this function returns the euclide distance between 2 vectors
    dist = sqrt(sum((P1 - P2).^ 2));   
end%-----------------------------------------------------------------------

function score = evaluate_difference(classes, testclasses)
    % score is the % of the correctly classified samples
    score = length(find(classes == testclasses));
    score = (score/length(classes))*100;  
end%-----------------------------------------------------------------------

function [takeout]=takeoutData(data,class,range,k)
    %analyze which samples have the largest distances with the samples of the same class
    number = (range(1,1)-1);%take the samples that are analyzing
    TClass = unique(class);%take just the classes that are analyzing
    LaC = length(TClass);
    I = zeros(k,LaC);%create a matrix for the index for each class for the k=number of samples that want to extract
    [euc_class] = Distance(data(:,1:2,range),data(:,1:2,range));
    [~,I] = maxk(euc_class(:,:),k);%equation for the max distances between points for k samples
    M = unique(I)'; %index of the samples that have the largest distance
    m = length(M);
    Count = zeros(1,m);
    for i = 1:m
        Count(:,i) = sum(sum(I(:,:)==M(:,i)));
    end
    MCount = [M;Count]';
    MCount = sortrows(MCount,2,'descend'); 
    takeout = MCount(1:k,1);
    takeout = takeout+number;
    takeout = takeout'; %Index of the samples that will be extract for the data
end%-----------------------------------------------------------------------

function [euc] = Distance(traindata,data)
%evaluate the distances between each sample of the same class
    euc = zeros(100,100);
    for i = 1:100
        for j = 1:100
           euc(i,j) = sum(diag(pdist2(traindata(:,:,i),data(:,:,j))));
        end
    end
end  %---------------------------------------------------------------------

function [scatteR] = plotscatter(sample,data,class)
%plot any sample of the data
    n = length(sample);
    for i=1:n
        e = sample(:,i);
        nexttile
        scatteR = scatter(data(:,1,e),data(:,2,e));
        c = class(:,e);
        title(['Class ',num2str(c)])
    end
end%-----------------------------------------------------------------------

function [inside1] = checkif(sample,Rows,data)
    %checks if the points are inside the shape
    x1 = [-1 -1 1 1]; %parameters to identify zero
    y1 = [-1.1 .2 .2 -1.1]; %parameters to identify zero
    polyin = polyshape({x1},{y1}); %shape that is create to 
    n = length(sample);
    in = zeros(Rows,n);
    for i = 1:n
        e = sample(:,i); 
        for j = 1:Rows   
            x = data(j,1,e);
            y = data(j,2,e);
            in(j,i) = isinterior(polyin,x,y);% assess whether or not there are points inside the shape
        end
    end
    inside1 = sum(in); %sum the points inside the shape if is zero then the class is zero, if is more than zero then the class is other.
end%-----------------------------------------------------------------------
