function [loss, h, y]  = rnn_forward(x, h0, d, Wxh, Whh, Why, bh, by)
    loss = 0;
    H = size(h0, 1);
    m = size(d, 1);
    bsize = size(x,2);
    y = zeros(m, bsize);
    h = zeros(H, bsize);
    for i = 1:bsize
        if i==1
            h(:,i) = tanh(Whh*h0 + Wxh*x(:,i) + bh);
        else
            h(:,i) = tanh(Whh*h(:,i-1) + Wxh*x(:,i) + bh);
        end
        y(:,i) = Why*h(:,i) + by;
        ei = d(:,i) - y(:,i);
        loss = loss + 0.5*sum(ei.^2);
    end    
end