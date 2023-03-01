tic()
load('MNIST_dataset');

n = 4000;
lambda = 10^(-7);

% get training subset
IxTrain = unique(randi(size(training.images,2),n,1));


% run for linear AF
num_reps = 1;
errorEvolLinearAll = [];
for repIx = 1:num_reps
    Nset= [ [1:500:7500] , [8000:250:10000] , [10250:1000:15000] ];
    Nset = Nset(randperm(length(Nset))); 
    errorEvolLinear = zeros(length(Nset),1);
    for Nix = 1:length(Nset)
        N = Nset(Nix);
        disp(100*Nix/length(Nset));
        error = getError(n,N,lambda,@(x) x,training,test,IxTrain);
        errorEvolLinear(Nix) = error;
    end
    [~,b] = sort(Nset);
    Nset = Nset(b);
    errorEvolLinear = errorEvolLinear(b);
    errorEvolLinearAll = [errorEvolLinearAll , errorEvolLinear]; 
    save('test')
end

% run for ReLU AF
errorEvolReLUAll = [];
for repIx = 1:num_reps
    Nset= [ [1:500:7500] , [8000:250:10000] , [10250:1000:15000] ];
    Nset = Nset(randperm(length(Nset))); 
    errorEvolReLU = zeros(length(Nset),1);
    paramEvol = cell(length(Nset));
    for Nix = 1:length(Nset)
        N = Nset(Nix);
        disp(100*Nix/length(Nset));
        error = getError(n,N,lambda,@(x) max(0,x),training,test,IxTrain);
        errorEvolReLU(Nix) = error;
    end

    [~,b] = sort(Nset);
    Nset = Nset(b);
    errorEvolReLU = errorEvolReLU(b);
    errorEvolReLUAll = [errorEvolReLUAll , errorEvolReLU]; 
    save('test')
end


% run for optimal linear AF
Nset= [ [1:500:7500] , [8000:250:10000] , [10250:1000:15000] ];
Nset = Nset(randperm(length(Nset))); 

errorEvolOpt = zeros(length(Nset),1);
for Nix = 1:length(Nset)
    N = Nset(Nix);
    disp(100*Nix/length(Nset));
    [error  ]= opterrorlinear(n,N,lambda,training,test,IxTrain);
    errorEvolOpt(Nix) = error;
    save('test')
end

[~,b] = sort(Nset);
Nset = Nset(b);
errorEvolOpt = errorEvolOpt(b);

% make plot
close all;
FigID = figure;
plot(Nset/4000   ,   mean(errorEvolLinearAll,2)    );
hold on; plot(Nset/4000   ,   mean(errorEvolReLUAll,2)  );
hold on;plot(Nset/4000,errorEvolOpt)
% change scale to log scale
set(gca, 'YScale', 'log')
toc()
%% auxiliary functions

function [error , paramout] = opterrorlinear(n,N,lambda,training,test,IxTrain)
   
    errorpoly = @(param) getError(n,N,lambda,@(x) (param.a)+(param.b).*x,training,test,IxTrain);

    a = optimizableVariable('a',[-2,2]);
    b = optimizableVariable('b',[-2,2]);
    
    results = bayesopt(errorpoly,[a,b],'MaxObjectiveEvaluations' ,40,'NumSeedPoints',4,'Verbose',0);
    
    error = results.MinObjective;
    paramout = results.XAtMinObjective;
    
end


function error = getError(n,N,lambda,sigma,training,test,IxTrain)

    d = 28*28;

    Theta = randn(N , d );
    Theta = (1./vecnorm(Theta'))'.*Theta;
    
    X = (sigma(Theta*training.images))';
    Y = -5 + training.labels/9;

    XTrain = X(IxTrain,:);
    YTrain = Y(IxTrain);
    
    XTest = (sigma(Theta*test.images))';
    YTest = -5 + test.labels/9;
    
    Mdl = fitrlinear(XTrain,YTrain,'Lambda',lambda,'Learner','leastsquares','Regularization','ridge','FitBias' ,true);

    error = sum((YTest - Mdl.predict(XTest)).^2)/length(YTest);
end



