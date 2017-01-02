% % % % % % % % % % % %
% PR Project 2
% Feature Selection
% author: Tmn07
% date: 2017-01-02 
% % % % % % % % % % % %

% 读取训练集数据
data = load('train.txt');
X = data(:,2:end);
y = data(:,1);

% 划分10折交叉验证
c = cvpartition(y,'k',10);

% 用SVM分类，构造目标函数
svmfun = @(xtrain,ytrain,xtest,ytest)...
	sum(svmclassify(svmtrain(xtrain,  ytrain),xtest ) ~= ytest);


% 特征选择
opts = statset('display','iter','TolFun',0); 
[fs,history] = sequentialfs(svmfun,X,y,'cv',c,'options',opts,'direction','forward');
% [fs,history] = sequentialfs(svmfun,X,y,'cv',c,'options',opts,'direction','backward');

% 用选择后的特征训练SVM
svmStruct = svmtrain(X(:,fs),y);

% 测试集数据
data2 = load('test.txt');
X2 = data2(:,2:end);
y2 = data2(:,1);

% 分类
C = svmclassify(svmStruct,X2(:,fs));

% 计算错误率等
errRate = sum(y2~= C)/length(y2)
Rate = 1 - errRate