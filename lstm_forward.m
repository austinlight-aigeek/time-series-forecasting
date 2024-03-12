function [hend, cend, y] = lstm_forward(...
    Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,inputs, h0, c0)

T = size(inputs, 2);        % length of LSTM network
D = size(V, 1);   % number of output nodes

y = zeros(D, T);

ht = h0;
ct = c0;

for j = 1:T
    xt = inputs(:, j);
    
    ft = logsig(Wf*xt + Rf*ht + bf);
    it = logsig(Wi*xt + Ri*ht + bi);
    gt = tanh(Wg*xt + Rg*ht + bg);
    ot = logsig(Wo*xt + Ro*ht + bo);
    
    ct = ft.*ct + it.*gt;
    ht = ot.*tanh(ct);
    yt = V*ht + b;
    
    y(:,j) = yt;
end

hend = ht;
cend = ct;

end
