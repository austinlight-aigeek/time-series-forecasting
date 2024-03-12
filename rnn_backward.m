function [dWxh,dWhh,dWhy,dbh,dby] = rnn_backward(x,h,y,d,Wxh,Whh,Why,bh,by)

    bsize = size(x,2);
    H = size(h,1);
    
    % derivatives for weights and biases
    dWxh = zeros(size(Wxh));
    dWhh = zeros(size(Whh));
    dWhy = zeros(size(Why));
    dbh = zeros(size(bh));
    dby = zeros(size(by));
    
    dh = zeros(H, bsize+1);         % for dL/dh(t)
    h = [h zeros(H,1)];    % removing h0 and adding h(bsize+1)
    
    for i=bsize:-1:1
        
        dy = -(d(:,i) - y(:,i));
        dWhy = dWhy + dy*h(:,i)';
        dby = dby + dy;

        dh(:,i) = Why'*dy + Whh'*dh(:,i+1).*(1-h(:,i+1).^2);
        dWxh = dWxh + dh(:,i).*(1-h(:,i).^2)*x(:,i)';
        if i>1
            dWhh = dWhh + dh(:,i).*(1-h(:,i).^2)*h(:,i-1)';
        end
        dbh = dbh + dh(:,i).*(1-h(:,i).^2);
    end
    
    dWxh = dWxh/bsize;
    dWhh = dWhh/bsize;
    dWhy = dWhy/bsize;
    dbh = dbh/bsize;
    dby = dby/bsize;
    
    % cliping gradient to avoid gradient explosion and vanishing
    dWxh = min(max(dWxh, -5), 5);
    dWhh = min(max(dWhh, -5), 5);
    dWhy = min(max(dWhy, -5), 5);
    dbh = min(max(dbh, -5), 5);
    dby = min(max(dby, -5), 5);
end