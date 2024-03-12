% time-series forecasting with rnn
clc
clear all
close all

load('Shanghai_Gold_Fix_PM.mat');

E = mean(data);
mu = std(data);
data = (data - E)/mu;

data_size = numel(data);
train_size = floor(data_size*0.8);
test_size = numel(data) - train_size;
train_data = data(1:train_size);
test_data = data(train_size+1:end);

m = 1;          % number of dates to be used to predict
l = 128;        % number of hidden nodes
pred = 1;       % prediction of forecast
bsize = 64;

if mod(train_size-m, bsize*pred)==0
    iter = (train_size - m)/bsize/pred;
    lastNumBatch = bsize;
else
    iter = ceil((train_size-m)/bsize/pred);
    lastNumBatch = floor((train_size-m-(iter-1)*bsize*pred)/pred);
    if lastNumBatch==0
        lastNumBatch = bsize;
        iter = iter-1;
    end
end

Wf = 0.01*randn(l,m);   Rf = 0.01*randn(l,l);   bf = 0.01*randn(l,1);
Wi = 0.01*randn(l,m);   Ri = 0.01*randn(l,l);   bi = 0.01*randn(l,1);
Wg = 0.01*randn(l,m);   Rg = 0.01*randn(l,l);   bg = 0.01*randn(l,1);
Wo = 0.01*randn(l,m);   Ro = 0.01*randn(l,l);   bo = 0.01*randn(l,1);
V = 0.01*randn(m,l);    b = 0.01*randn(m,1);

mWf = zeros(l,m);   mRf = zeros(l,l);   mbf = zeros(l,1);
mWi = zeros(l,m);   mRi = zeros(l,l);   mbi = zeros(l,1);
mWg = zeros(l,m);   mRg = zeros(l,l);   mbg = zeros(l,1);
mWo = zeros(l,m);   mRo = zeros(l,l);   mbo = zeros(l,1);
mV = zeros(m,l);    mb = zeros(m,1);

epoch_num = 200;
learning_rate = 0.0001;
mnt_rate = 0.9;

for epoch = 1:epoch_num
    h0 = zeros(l,1);
    c0 = zeros(l,1);
    L = 0;
    
    for i = 1:iter
        inputs = [];
        targets = [];
        
        if i~=iter
            for j = 1:bsize
                s1 = (i-1)*bsize*pred+(j-1)*pred+1;
                e1 = s1 + m -1;
                s2 = s1 + pred;
                e2 = s2 + m -1;
                inputs = [inputs train_data(s1:e1)'];
                targets = [targets train_data(s2:e2)'];
            end
        else
            for j = 1:lastNumBatch
                s1 = (i-1)*bsize*pred+(j-1)*pred+1;
                e1 = s1 + m -1;
                s2 = s1 + pred;
                e2 = s2 + m -1;
                inputs = [inputs train_data(s1:e1)'];
                targets = [targets train_data(s2:e2)'];
            end
        end
        
        [dWf,dRf,dbf,dWi,dRi,dbi,dWg,dRg,dbg,dWo,dRo,dbo,dV,db, h0, c0, loss] = ...
            lstm(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,inputs, targets, h0, c0);
        
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
    
    if (~mod(epoch, 10) || epoch == 1)
        str = sprintf('epoch: %d, loss: %f', epoch, L);
        disp(str);
    end
end

h0 = zeros(l,1);
c0 = zeros(l,1);

Btotal = floor((data_size-m)/pred)+1;
y_hat = [];
for i = 1:Btotal
    s1 = (i-1)*pred + 1;
    e1 = s1 + m -1;
    inputs = data(s1:e1)';
    [h0, c0, yy] = lstm_forward(...
        Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,inputs, h0, c0);
    y_hat = [y_hat yy(end-pred+1: end)];
end

y_hat = y_hat*mu + E;
y_pred = y_hat(train_size-m+1:end-1);
targets = test_data*mu + E;

plot(targets);
hold on;
plot(y_pred);

%RMSE = sqrt(mean((targets - y_pred).^2))




