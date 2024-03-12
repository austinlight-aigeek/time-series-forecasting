clear all
close all
clc

load('Shanghai_Gold_Fix_PM.mat');
input_data = data';
E = mean(input_data);
sd = std(input_data);

input_data = (input_data-E)/sd;
data_size = length(input_data);             % total data size

div = 0.8;      % rate of amount of train data size

train_size = floor(data_size*div);          % train data size

m = 30;         % input unit size (# of xt)
H = 256;         % hidden unit size (# of ht)
n = 30;         % output unit size (# of yt)
pred = 5;       % prediction step size

% preparing data for training and test
train_size = train_size - mod(train_size-n, pred);
data_size = data_size - mod(data_size-n, pred);

input_data = input_data(1:data_size);
train_data = input_data(1:train_size);      % training data
test_data = input_data(train_size+1:end);

% weights and biases, and momentum
Wxh = randn(H,n)*0.01; mWxh = zeros(size(Wxh));
Whh = randn(H,H)*0.01; mWhh = zeros(size(Whh));
Why = randn(m,H)*0.01; mWhy = zeros(size(Why));
bh = randn(H,1)*0.01; mbh = zeros(size(bh));
by = randn(m,1)*0.01; mby = zeros(size(by));

h0 = zeros(H, 1);       % h0 initialization

learning_rate = 0.001;
m_rate = 0.9;           % momentum rate
epoch_num = 300;

bsize = 20;             % mini-batch size
blist = 1:bsize*pred:train_size;    % mini-batch first index list

loss = 0;
% figure(1);
% ge = animatedline;
% ge.Color = [0 0.4470 0.7410];
% addpoints(ge,0,loss);
% drawnow;
% xlim([0 epoch_num]);
% ylim([0 200]);
% xlabel('epoch'); ylabel('total loss');

h_train = figure(2);
h_train.Position = [800 260 560 420];
xx = 1:train_size;
train_data_original = train_data * sd + E;
g_train = plot(xx, train_data_original,xx,zeros(size(train_data)));
axis([1, train_size, min(train_data_original)-5, max(train_data_original)+5]);
title('Training procedure');

for epoch = 1:epoch_num
    loss = 0;               % total loss
    for p = blist
        % check if full mini-batch possible
        if (p + pred*bsize +n - 1<= train_size)
            ibsize = bsize;
        else
            ibsize = floor((train_size - p - n + 1)/pred);
        end
        
        % prepare input-target data for forward pass
        x = zeros(n, ibsize);
        d = zeros(m, ibsize);
        for i = 1:ibsize
            pp = p+(i-1)*pred;
            x(:,i) = train_data(pp:pp+n-1)';
            d(:,i) = train_data(pp+pred:pp+n+pred-1)';
        end
        
        [L, h, y] = rnn_forward(x, h0, d, Wxh, Whh, Why, bh, by);
        [dWxh,dWhh,dWhy,dbh,dby] = rnn_backward(x,h,y,d,Wxh,Whh,Why,bh,by);
        
        mWxh = learning_rate*dWxh + m_rate*mWxh;
        mWhh = learning_rate*dWhh + m_rate*mWhh;
        mWhy = learning_rate*dWhy + m_rate*mWhy;
        mbh = learning_rate*dbh + m_rate*mbh;
        mby = learning_rate*dby + m_rate*mby;
        
        Wxh = Wxh - mWxh;
        Whh = Whh - mWhh;
        Why = Why - mWhy;
        bh = bh - mbh;
        by = by - mby;
        loss = loss + L;
    end
    
    if ~mod(epoch, 10)
        y_hat = zeros(size(train_data));
        for p = blist
            % check if full mini-batch possible
            if (p + pred*bsize + n - 1<= train_size)
                ibsize = bsize;
            else
                ibsize = floor((train_size - p - n + 1)/pred);
            end
            % prepare input-target data for forward pass
            x = zeros(n, ibsize);
            d = zeros(m, ibsize);
            for i = 1:ibsize
                pp = p+(i-1)*pred;
                x(:,i) = train_data(pp:pp+n-1)';
                d(:,i) = train_data(pp+pred:pp+n+pred-1)';
            end
            [L, h, y] = rnn_forward(x, h0, d, Wxh, Whh, Why, bh, by);
            for i = 1:ibsize
                yy = y(:,i);
                y_hat(p+(i-1)*pred+n:p+i*pred+n-1) = yy(end-pred+1:end);
            end
        end
        y_hat(1:n) = train_data(1:n);
        y_hat_original = y_hat*sd+E;
        g_train(2).YData = y_hat_original;
        drawnow;
        if (mod(epoch, 20)==0)        
            [MAE, MAPE] = eval_error(train_data_original, y_hat_original);
            txt = strcat('epoch: ',num2str(epoch),' loss_func: ',num2str(loss));
            txt = strcat(txt, ' MAE: ', num2str(MAE), ' MAPE: ', num2str(MAPE),'%');
            disp(txt);
        end
    end
end


h_test = figure(3);
h_test.Position = [800 40 560 420];
xx = 1:data_size;
input_data_original = input_data * sd + E;
g_test = plot(xx, input_data_original ,xx, zeros(size(input_data)));
axis([train_size+1, data_size, min(input_data_original)-5, max(input_data_original)+5]);
title('Predicted Result');
loss = 0;
blist = 1:bsize*pred:data_size;    % mini-batch first index list
y_hat = zeros(data_size, 1);
for p = blist
    if (p + pred*bsize + n - 1<= data_size)
        ibsize = bsize;
    else
        ibsize = floor((data_size - p - n + 1)/pred);
    end
    
    % prepare input-target data for forward pass
    x = zeros(n, ibsize);
    d = zeros(m, ibsize);
    for i = 1:ibsize
        pp = p+(i-1)*pred;
        x(:,i) = input_data(pp:pp+n-1)';
        d(:,i) = input_data(pp+pred:pp+n+pred-1)';
    end
    [L, h, y] = rnn_forward(x, h0, d, Wxh, Whh, Why, bh, by);
    loss = loss + L;
    for i = 1:ibsize
        yy = y(:,i);
        y_hat(p+(i-1)*pred+n:p+i*pred+n-1) = yy(end-pred+1:end);
    end
end
y_hat(1:n) = input_data(1:n);
y_hat_original = y_hat * sd + E;
g_test(2).YData = y_hat_original;

[MAE, MAPE] = eval_error(input_data_original(train_size+1:end), y_hat_original(train_size+1:end));
txt = strcat('loss_func: ', num2str(loss), ' MAE: ', num2str(MAE), ' MAPE: ', num2str(MAPE),'%');
disp('********************************************');
disp(txt);

disp('==========================================================');


