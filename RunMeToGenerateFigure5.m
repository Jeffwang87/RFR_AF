load('MNIST_dataset.mat');

d = 28*28;
psi2 = 10;
n = psi2 * d;
N = 10000;

num_reps = 5;
errorEvolReLUAll = [];
for repIx = 1:num_reps
    LambdaSet = 10.^[-2:0.2:0];
    errorEvolReLU = zeros(length(LambdaSet),1);
    for lambda_ix = 1:length(LambdaSet)
        disp([repIx , 100*lambda_ix /length(LambdaSet) ]);
        lambda = LambdaSet(lambda_ix);
        error = getError(n,N,lambda,@(x) max(0,x),training,test,false,false);    
        errorEvolReLU(lambda_ix) = error;
    end
    errorEvolReLUAll = [errorEvolReLUAll  , errorEvolReLU];
end
%%
LambdaSet = 10.^[-2:0.2:0];

errorEvolQuad = zeros(length(LambdaSet),1);
for lambda_ix = 1:length(LambdaSet)
    disp(100*lambda_ix /length(LambdaSet) );
    lambda = LambdaSet(lambda_ix);

    [errorQ , ~ ]= opterrorquad(n,N,lambda,training,test,false,false);
    errorEvolQuad(lambda_ix) = errorQ;
end

close all;
fID = figure; 
plot(  log10(LambdaSet)   ,   mean(errorEvolReLUAll, 2) );
hold on;
plot(log10(LambdaSet),errorEvolQuad)



%% auxiliary functions

function [error , paramout] = opterrorquad(n,N,lambda,training,test,normalizeFlag,BiasFlag)

    errorpoly = @(param) getError(n,N,lambda,@(x) (param.a)+(param.b).*x+(param.c).*x.*x,training,test,normalizeFlag,BiasFlag);

    a = optimizableVariable('a',[-2,2]);
    b = optimizableVariable('b',[-2,2]);
    c = optimizableVariable('c',[-2,2]);
    
    results = bayesopt(errorpoly,[a,b,c],'MaxObjectiveEvaluations' ,50,'NumSeedPoints',4,'Verbose',0);
    
    error = results.MinObjective;
    paramout = results.XAtMinObjective;
    
end


function error = getError(n,N,lambda,sigma,training,test,normalizeFlag,BiasFlag)

    d = 28*28;

    Theta = randn(N , d );
    Theta = (1./vecnorm(Theta'))'.*Theta;
    
    X = (sigma(Theta*training.images))';

    Y = (training.labels/9)-0.5;

    IxTrain = unique(randi(size(training.images,2),n,1));
    
    XTrain = X(IxTrain,:);
    YTrain = Y(IxTrain);
    
    if (normalizeFlag == true)
        XTrain = normalize(XTrain);
    end

    XTest = (sigma(Theta*test.images))';
    YTest = (test.labels/9)-0.5;

    if (normalizeFlag == true)
        XTest = normalize(XTest);
    end
    
    Mdl = fitrlinear(XTrain,YTrain,'Lambda',lambda,'Learner','leastsquares','Regularization','ridge','FitBias' ,BiasFlag);

    error = sum((YTest - Mdl.predict(XTest)).^2)/length(YTest);
end

