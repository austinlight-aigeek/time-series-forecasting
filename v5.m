clc
clear all
close all

m = 30;         % number of data to be used for forecasting
pred = 5;       % number of prediction
bsize = 10;     % batch size
H = 128;        % dimension of hidden layer
T = 5;          % length of LSTM network

epochNum = 300;

eta = 0.0001;
mRate = 0.9;
dispRate = 10;

load('Shanghai_Gold_Fix_PM.mat');

div = 0.8;
dataSize = length(data) - mod(length(data), pred);
trainSize = floor(dataSize*div);
trainSize = trainSize - mod(trainSize, pred);
testSize = dataSize - trainSize;

data = data(1:dataSize);

% calculation of batch iteration and last batch size
[numBatch, lastBSize] = calLastBSize(trainSize, pred, bsize*T, m);

trainDataOriginal = data(1:trainSize);
testDataOriginal = data(trainSize+1:end);

mu = mean(trainDataOriginal);
sig = std(trainDataOriginal);

trainData = (trainDataOriginal - mu)/sig;
testData = (testDataOriginal - mu)/sig;

% =========================================================================
% weights and biases of LSTM
Wf = 0.01*randn(H,m);   Rf = 0.01*randn(H,H);   bf = 0.01*randn(H,1);
Wi = 0.01*randn(H,m);   Ri = 0.01*randn(H,H);   bi = 0.01*randn(H,1);
Wg = 0.01*randn(H,m);   Rg = 0.01*randn(H,H);   bg = 0.01*randn(H,1);
Wo = 0.01*randn(H,m);   Ro = 0.01*randn(H,H);   bo = 0.01*randn(H,1);
V = 0.01*randn(pred,H);    b = 0.01*randn(pred,1);

mWf = zeros(H,m);       mRf = zeros(H,H);       mbf = zeros(H,1);
mWi = zeros(H,m);       mRi = zeros(H,H);       mbi = zeros(H,1);
mWg = zeros(H,m);       mRg = zeros(H,H);       mbg = zeros(H,1);
mWo = zeros(H,m);       mRo = zeros(H,H);       mbo = zeros(H,1);
mV = zeros(pred,H);     mb = zeros(pred,1);
% =========================================================================

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
    
    h0 = zeros(H,1);
    c0 = zeros(H,1);
    
    epochLoss = 0;
    yTrainPred = [];
    for i = 1:numBatch
        % check if it is the last iteration of batch
        if i ~= numBatch
            B = bsize;
        else
            B = floor(lastBSize/T);
        end
        
        dWf = zeros(H,m);       dRf = zeros(H,H);       dbf = zeros(H,1);
        dWi = zeros(H,m);       dRi = zeros(H,H);       dbi = zeros(H,1);
        dWg = zeros(H,m);       dRg = zeros(H,H);       dbg = zeros(H,1);
        dWo = zeros(H,m);       dRo = zeros(H,H);       dbo = zeros(H,1);
        dV = zeros(pred,H);     db = zeros(pred,1);
        
        L = 0;
        yy = [];
        for j = 1:B
            if (i==numBatch) && (j==B)
                TT = mod(lastBSize, T);
            else
                TT = T;
            end

            sx = (j-1)*T*pred+(i-1)*T*pred*bsize+1;
            ex = sx + m - 1;
            sy = ex + 1;
            ey = sy + pred - 1;
            XTrain = zeros(m, T);
            YTrain = zeros(pred, T);
            for k = 1:TT
                ssx = sx + (k-1)*pred;
                eex = ex + (k-1)*pred;
                ssy = sy + (k-1)*pred;
                eey = ey + (k-1)*pred;

                XTrain(:,k) = trainData(ssx:eex);
                YTrain(:,k) = trainData(ssy:eey);
            end
                
            [ddWf,ddRf,ddbf,ddWi,ddRi,ddbi, ...
                ddWg,ddRg,ddbg,ddWo,ddRo,ddbo, ...
                ddV,ddb, h0, c0, loss, y] = ...
                lstm(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,XTrain, YTrain, h0, c0);
            
            yy = [yy y(:)'];
            
            L = L + loss;
            dWf = dWf + ddWf;   dRf = dRf + ddRf;   dbf = dbf + ddbf;
            dWi = dWi + ddWi;   dRi = dRi + ddRi;   dbi = dbi + ddbi;
            dWg = dWg + ddWg;   dRg = dRg + ddRg;   dbg = dbg + ddbg;
            dWo = dWo + ddWo;   dRo = dRo + ddRo;   dbo = dbo + ddbo;
            dV = dV + ddV;  db = db + ddb;
        end
        
        yTrainPred = [yTrainPred yy];
        dWf = dWf/bsize;	dRf = dRf/bsize;    dbf = dbf/bsize;
        dWi = dWi/bsize;	dRi = dRi/bsize;    dbi = dbi/bsize;
        dWg = dWg/bsize;	dRg = dRg/bsize;    dbg = dbg/bsize;
        dWo = dWo/bsize;	dRo = dRo/bsize;    dbo = dbo/bsize;
        dV = dV/bsize;  db = db/bsize;
        
        mWf = mRate*mWf - eta*dWf;
        mWi = mRate*mWi - eta*dWi;
        mWg = mRate*mWg - eta*dWg;
        mWo = mRate*mWo - eta*dWo;
        
        mRf = mRate*mRf - eta*dRf;
        mRi = mRate*mRi - eta*dRi;
        mRg = mRate*mRg - eta*dRg;
        mRo = mRate*mRo - eta*dRo;
        
        mbf = mRate*mbf - eta*dbf;
        mbi = mRate*mbi - eta*dbi;
        mbg = mRate*mbg - eta*dbg;
        mbo = mRate*mbo - eta*dbo;
        
        mV = mRate*mV - eta*dV;
        mb = mRate*mb - eta*db;
        
        Wf = Wf + mWf;  Rf = Rf + mRf; bf = bf + mbf;
        Wi = Wi + mWi;  Ri = Ri + mRi; bi = bi + mbi;
        Wg = Wg + mWg;  Rg = Rg + mRg; bg = bg + mbg;
        Wo = Wo + mWo;  Ro = Ro + mRo; bo = bo + mbo;
        V = V + mV;     b = b + mb;
        
        epochLoss = epochLoss + L;
        
    end
    
    if ~mod(epoch, dispRate)
%         g1(2).YData = yTrainPred*sig + mu;
%         drawnow;
        str = strcat('epoch: ', num2str(epoch), ' loss: ', num2str(epochLoss));
        disp(str);
    end
end
% ======================== training completed =============================


% calculation of batch iteration and last batch size
[numBatch, lastBSize] = calLastBSize(testSize, pred, bsize*T, m);

yTestPred = [];
for i = 1:numBatch
    % check if it is the last iteration of batch
    if i ~= numBatch
        B = bsize;
    else
        B = floor(lastBSize/T);
    end
    
    L = 0;
    yy = [];
    for j = 1:B
        if (i==numBatch) && (j==B)
            TT = mod(lastBSize, T);
        else
            TT = T;
        end
        
        sx = (j-1)*T*pred+(i-1)*T*pred*bsize+1;
        ex = sx + m - 1;
        sy = ex + 1;
        ey = sy + pred - 1;
        XTrain = zeros(m, T);
        YTrain = zeros(pred, T);
        for k = 1:TT
            ssx = sx + (k-1)*pred;
            eex = ex + (k-1)*pred;
            ssy = sy + (k-1)*pred;
            eey = ey + (k-1)*pred;
            
            XTrain(:,k) = trainData(ssx:eex);
            YTrain(:,k) = trainData(ssy:eey);
        end
        
        [~,~,~,~,~,~,~,~,~,~,~,~,~,~, h0, c0, loss, y] = ...
            lstm(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,XTrain, YTrain, h0, c0);
        
        yy = [yy y(:)'];
        L = L + loss;
    end
    
    yTestPred = [yTestPred yy];
end

% visualizing test result
h2 = figure(2);
h2.Position = [10 60 600 280];
testTarget = testDataOriginal(m+1:end-pred-10);
plot(testTarget);
hold on
testPred = yTestPred(1:end)*sig + mu;
plot(testPred);
hold off;

[RMSE, MAPE] = eval_error(testTarget, testPred);
str = sprintf('RMSE: %f, MAPE: %f', RMSE, MAPE);
disp(str);

