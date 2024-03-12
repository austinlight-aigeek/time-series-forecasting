clc
clear all
close all

load Shanghai_Gold_Fix_PM

m = 1;                    % input dimension
l = 32;                  % hidden node dimension
n = 1;                    % target dimension
miniBatchSize = 20;       % length of LSTM network

% Partition the training and test data

data_size = numel(data);
train_size = floor(numel(data)*0.9/miniBatchSize/m)*miniBatchSize*m+m;
test_size = numel(data)-train_size;

train_data = data(1:train_size);
test_data = data(train_size+1:end);

% Standardize Data
mu = mean(train_data);
sigma = std(train_data);

train_data = (train_data-mu)/sigma;
test_data = (test_data-mu)/sigma;

denoised_train_data = wdenoise(train_data, 3,'Wavelet','db4',...
    'DenoisingMethod','SURE');

% plot(train_data,'r-')
% hold on
% plot(denoised_train_data,'b-')
% legend('Train Data','Denoised Train Data')
% hold off

Wf = 0.01*randn(l,m);   Rf = 0.01*randn(l,l);   bf = 0.01*randn(l,1);
Wi = 0.01*randn(l,m);   Ri = 0.01*randn(l,l);   bi = 0.01*randn(l,1);
Wg = 0.01*randn(l,m);   Rg = 0.01*randn(l,l);   bg = 0.01*randn(l,1);
Wo = 0.01*randn(l,m);   Ro = 0.01*randn(l,l);   bo = 0.01*randn(l,1);
V = 0.01*randn(n,l);    b = 0.01*randn(n,1);

mWf = zeros(l,m);       mRf = zeros(l,l);       mbf = zeros(l,1);
mWi = zeros(l,m);       mRi = zeros(l,l);       mbi = zeros(l,1);
mWg = zeros(l,m);       mRg = zeros(l,l);       mbg = zeros(l,1);
mWo = zeros(l,m);       mRo = zeros(l,l);       mbo = zeros(l,1);
mV = zeros(n,l);        mb = zeros(n,1);

epoch_num = 200;
learning_rate = 0.001;
mnt_rate = 0.9;

numMiniBatch = floor(train_size/miniBatchSize/m);   % number of possible full mini-batches
bList = 1:miniBatchSize:(numMiniBatch-1)*miniBatchSize+1;   % min-batch index list

desired_output = sigma*denoised_train_data(2:end)+mu;
h_train = figure(1);
h_train.Position = [800 260 560 420];
xx = 1:train_size-1;
g_train = plot(xx, desired_output, xx, zeros(size(desired_output)));
axis([1, train_size-1, min(desired_output)-5, max(desired_output)+5]);
title('Training procedure');

figure(2);
ge = animatedline;
title('Loss function');

L_temp = [];
for epoch = 1:epoch_num
    h0 = zeros(l,1);
    c0 = zeros(l,1);
    L = 0;

    if mod(epoch, 20)==0
        L_temp = [];
    end
    real_output = [];
    for p = 1:numMiniBatch
        bStart = (p-1)*miniBatchSize*m+1;
        
        input = [];
        target = [];
        for i = 1:miniBatchSize
            s = bStart + (i-1)*m;
            input = [input denoised_train_data(s:s+m-1)'];
            target = [target denoised_train_data(s+m:s+2*m-1)'];
        end
        
        [dWf,dRf,dbf,dWi,dRi,dbi,dWg,dRg,dbg,dWo,dRo,dbo,dV,db,h0,c0,loss, y_hat] = ...
            lstm(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,input,target,h0,c0);
        
        real_output = [real_output sigma*y_hat(:)'+mu];        
        
        mWf = mnt_rate*mWf - learning_rate*dWf;
        mWi = mnt_rate*mWi - learning_rate*dWi;
        mWg = mnt_rate*mWg - learning_rate*dWg;
        mWo = mnt_rate*mWo - learning_rate*dWo;
        
        mRf = mnt_rate*mRf - learning_rate*dRf;
        mRi = mnt_rate*mRi - learning_rate*dRi;
        mRg = mnt_rate*mRg - learning_rate*dRg;
        mRo = mnt_rate*mRo - learning_rate*dRo;
        
        mbf = mnt_rate*mbf - learning_rate*dbf;
        mbi = mnt_rate*mbi - learning_rate*dbi;
        mbg = mnt_rate*mbg - learning_rate*dbg;
        mbo = mnt_rate*mbo - learning_rate*dbo;
        
        mV = mnt_rate*mV - learning_rate*dV;
        mb = mnt_rate*mb - learning_rate*db;
        
        Wf = Wf + mWf;  Rf = Rf + mRf; bf = bf + mbf;
        Wi = Wi + mWi;  Ri = Ri + mRi; bi = bi + mbi;
        Wg = Wg + mWg;  Rg = Rg + mRg; bg = bg + mbg;
        Wo = Wo + mWo;  Ro = Ro + mRo; bo = bo + mbo;
        V = V + mV;     b = b + mb;
        
        L = L + loss;
    end
    
    L_temp = [L_temp L];

    
    g_train(2).YData = real_output;
    drawnow;
%     pause(0.5);
    
    addpoints(ge,epoch,L);
    drawnow;
    xlim([1 epoch_num]);
    if epoch > 40
        ylim([0 10]);
    else
        ylim([0 750]);
    end
    xlabel('epoch'); ylabel('Error');
    
    if (~mod(epoch, 10) || epoch == 1)
        str = sprintf('epoch: %d, loss: %f', epoch, L);
        disp(str);
    end
end

h0 = zeros(l,1);
c0 = zeros(l,1);
for p = 1:numMiniBatch
    bStart = (p-1)*miniBatchSize*m+1;
    input = [];
    for i = 1:miniBatchSize
        s = bStart+(i-1)*m;
        input = [input denoised_train_data(s:s+m-1)'];
        [h0,c0,yy] = lstm_forward(...
            Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,input,h0,c0);
    end
end

hend = h0;
cend = c0;
numBTest_size = floor(test_size/m);
pred = [];

for i = 1:numBTest_size
    input = test_data((i-1)*m+1:i*m)';
    [hend,cend,pp] = lstm_forward(...
        Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,input,hend,cend);
    pred = [pred pp'];
end

pred = pred*sigma + mu;
target = test_data*sigma + mu;

figure(3)
xx = 1:test_size;
plot(target(2:end));
hold on
plot(pred(1:end-1))
hold off

[RMSE, MAPE] = eval_error(target(2:end), pred(1:end-1));

str = sprintf('RMSE: %f, MAPE: %f', RMSE, MAPE);
disp(str);

