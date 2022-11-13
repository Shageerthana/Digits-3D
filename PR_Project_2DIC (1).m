% pca
% reducing resample
% remove farther outliers

% importing dataset if not already imported
% don't comment

dimensions = 2;
if ~(exist('data','var')) && ~(exist('class','var'))
    [data,class] = get_all_dataset(60,dimensions);
    % randomization to be checked
    % comment the lines if you don't want to randomize
%     rand_ind = randperm(size(class,2));
%     data = data(:,:,rand_ind);
%     class = class(rand_ind);
end
% add Daniela-----------------------------------
k=10;

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
takeout = [takeout0' takeout1' takeout2' takeout3' takeout4' takeout5' takeout6' takeout7' takeout8' takeout9'];
data(:,:,takeout) = [];
class(takeout) = [];
%----------------------------------------------------------------
  
rand_ind = randperm(size(class,2));
data = data(:,:,rand_ind);
class = class(rand_ind);

% dividing train and test-----------------------------------------
    t = 0.8;
    NUM = fix(size(data,3));
    traindata = data(:,:,1:NUM*t);
    trainclass = class(:,1:NUM*t);
    testdata = data(:,:,NUM*t+1:end);
    testclass = class(:,NUM*t+1:end);
%--------------------------------------------------------------------    

 
% plot first 10 samples
%     figure
%     for i = 1:10
%         subplot(2,5,i);
%         tmp = testdata(:,:,i);
%         tmp = myminmax(tmp);
%         scatter(tmp(:,1), tmp(:,2));
%     end


C = knn(trainclass, traindata, testdata, 11);
result = evaluate_difference(C, testclass);
confmat = confusionmat(testclass,C);
figure
confusionchart(confmat);
for i = 0:9
    for j = 0:9
        tmp0 = find(testclass==i & C==j);
        tmp(i+1,j+1) = length(tmp0);
    end
end

% mink = 2;
% maxk = 12;
% results = zeros(1, maxk-mink+1);

% evaluate k
% for i=maxk-mink+1:-1:1
%     C = knn(trainclass, traindata, testdata, i);
%     results(i) = evaluate_difference(C, testclass);
% end

% [m,mind] = min(results);
% fprintf("From %d to %d, the best result is obtained with k = %d\n", mink, maxk, mink+mind-1);


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
            r= size(tmp, 1);
            data(:,:,(i*100)+j) = resample(tmp, num_points, r, dim);
            % data(:,:,(i*100)+j) = reorder_sample(data(:,:,(i*100)+j), dim);
            % data(:,:,(i*100)+j) = myminmax(data(:,:,(i*100)+j));
        end
    end
end

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
    [n_nearest, index] = sort(distance_matrix);
    train = class(index);
    k_nearest = n_nearest(1:k,:);
    train = train(1:k,:);
    for i = 1:size(data,3)
        for j = 1:max(trainclass)
            tmp(j,i) = sum(train(:,i)==j);
        end
    end
    [~, C] = max(tmp);
end

function dist = sample_dist(S1, S2)
% this function returns the sum of the point by point distances between two
% samples. S1 ansd S2 are numpointsxdim matrices
    dist = 0;
    for i=1:size(S1,1)
        dist = dist + euc_distance(S1(1,:), S2(2,:));
    end
end

function dist = euc_distance(P1, P2)
% this function returns the euclide distance between 2 vectors
%     if size(P1,1) == size(P2,1) && size(P2,1) == 1
%         dist = zeros(1,size(P1,2));
%         for i = 1:size(P1,2)
%             dist(i) = sqrt((P1(i) - P2(i)).^ 2);
%         end
%     else
        dist = sqrt(sum((P1 - P2).^ 2));
%     end    
end

function sample = reorder_sample(S1, dim)
    sample = zeros(size(S1));
    [sample(:,1), ind_points] = sort(S1(:,1));
    for k = 2:dim
        tmp2 = S1(:,k);
        sample(:,k) = tmp2(ind_points);
    end
end

function score = evaluate_difference(classes, testclasses)
    % score is the number of misclassified samples
    score = length(find(classes == testclasses));
    score = (score/length(classes))*100;  
end

function scaled = myminmax(X)
% given matrix X returns the same matrix with the values scaled with
% min max scaling per column
    [r, c] = size(X);
    scaled = X;
    for i=1:c
        scaled(:,i) = (scaled(:,i) - ones([r,1])*min(scaled(:,i)))/(max(scaled(:,i)) - min(scaled(:,i)));
    end
end
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