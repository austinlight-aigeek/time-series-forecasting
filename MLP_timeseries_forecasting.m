clc
clear all
close all

m = 1;         % number of data to be used for forecasting
pred = 1;       % number of prediction
bsize = 20;     % batch size
H = 128;        % dimension of hidden layer
epochNum = 200;

load('Shanghai_Gold_Fix_PM.mat');

div = 0.8;
dataSize = length(data) - mod(length(data), pred);
trainSize = floor(dataSize*div);
trainSize = trainSize - mod(trainSize, pred);
testSize = dataSize - trainSize;

data = data(1:dataSize);

% calculation of batch iteration and last batch size
[numBatch, lastBSize] = calLastBSize(trainSize, pred, bsize, m);

trainDataOriginal = data(1:trainSize);
testDataOriginal = data(trainSize+1:end);

mu = mean(trainDataOriginal);
sig = std(trainDataOriginal);

trainData = (trainDataOriginal - mu)/sig;
testData = (testDataOriginal - mu)/sig;

% =========================================================================
% weights and biases of MLP
w1 = 0.01*randn(H, m);  b1 = 0.01*randn(H, 1);
w2 = 0.01*randn(H, H);  b2 = 0.01*randn(H, 1);
w3 = 0.01*randn(pred, H);   b3 = 0.01*randn(pred, 1);

mw1 = zeros(size(w1));  mw2 = zeros(size(w2));  mw3 = zeros(size(w3));
mb1 = zeros(size(b1));  mb2 = zeros(size(b2));  mb3 = zeros(size(b3));
% =========================================================================

eta = 0.001;
mRate = 0.9;
dispRate = 10;

% ======================== start training =================================
% visualizing training procedure
h1 = figure(1);
h1.Position = [10 400 600 280];
t = 1:trainSize-pred;
tp = m+1:trainSize;
g1 = plot(t, trainDataOriginal(1:end-pred), tp, zeros(size(tp)));
ylim([min(trainDataOriginal)*0.95, max(trainDataOriginal)*1.05]);
title('Training Procedure');
l1 = legend('Original Value', 'Predicted Value');
l1.Position = [0.58 0.76 0.22 0.13];

for epoch = 1:epochNum
    epochLoss = 0;
    yTrainPred = [];
    for i = 1:numBatch
        % check if it is the last iteration of batch
        if i ~= numBatch
            B = bsize;
        else
            B = lastBSize;
        end
        
        % batch training data construction
        XTrain = zeros(m, B);
        YTrain = zeros(pred, B);
        for j = 1:B
            sx = (i-1)*bsize*pred + (j-1)*pred + 1;
            ex = sx + m -1;
            sy = ex + 1;
            ey = sy + pred - 1;
            XTrain(:,j) = trainData(sx:ex);
            YTrain(:,j) = trainData(sy:ey);
        end
        
        [dw1, dw2, dw3, db1, db2, db3, loss, y] = ...
            predMLP(XTrain, YTrain, w1, w2, w3, b1, b2, b3);
        yTrainPred = [yTrainPred y(:)'];
        
        mw1 = mRate*mw1 - eta*dw1;
        mw2 = mRate*mw2 - eta*dw2;
        mw3 = mRate*mw3 - eta*dw3;
        mb1 = mRate*mb1 - eta*db1;
        mb2 = mRate*mb2 - eta*db2;
        mb3 = mRate*mb3 - eta*db3;
        
        w1 = w1 + mw1;  w2 = w2 + mw2;  w3 = w3 + mw3;
        b1 = b1 + mb1;  b2 = b2 + mb2;  b3 = b3 + mb3;
        
        epochLoss = epochLoss + loss;
        
    end
    
    if ~mod(epoch, dispRate)
        g1(2).YData = yTrainPred*sig + mu;
        drawnow;
        str = strcat('epoch: ', num2str(epoch), ' loss: ', num2str(epochLoss));
        disp(str);
    end
end
% ======================== training completed =============================


% calculation of batch iteration and last batch size
[numBatch, lastBSize] = calLastBSize(testSize, pred, bsize, m);

yTestPred = [];
for i = 1:numBatch
    % check if it is the last iteration of batch
    if i ~= numBatch
        B = bsize;
    else
        B = lastBSize;
    end
    
    % batch training data construction
    XTrain = zeros(m, B);
    YTrain = zeros(pred, B);
    for j = 1:B
        sx = (i-1)*bsize*pred + (j-1)*pred + 1;
        ex = sx + m -1;
        sy = ex + 1;
        ey = sy + pred - 1;
        XTrain(:,j) = testData(sx:ex);
        YTrain(:,j) = testData(sy:ey);
    end
    
    [~, ~, ~, ~, ~, ~, ~, y] = ...
        predMLP(XTrain, YTrain, w1, w2, w3, b1, b2, b3);
    
    yTestPred = [yTestPred y(:)'];
end

% visualizing test result
h2 = figure(2);
h2.Position = [10 60 600 280];
testTarget = testDataOriginal(m+1:end-pred);
plot(testTarget);
hold on
testPred = yTestPred(1:end-pred)*sig + mu;
plot(testPred);
hold off;

[RMSE, MAPE] = eval_error(testTarget, testPred);
str = sprintf('RMSE: %f, MAPE: %f', RMSE, MAPE);
disp(str);

