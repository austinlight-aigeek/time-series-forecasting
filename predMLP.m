function [dw1, dw2, dw3, db1, db2, db3, loss, y] = ...
    predMLP(x, d, w1, w2, w3, b1, b2, b3)

% number of batches
bsize = size(x, 2);
pred = size(d, 1);

dw1 = zeros(size(w1));  dw2 = zeros(size(w2));  dw3 = zeros(size(w3));
db1 = zeros(size(b1));  db2 = zeros(size(b2));  db3 = zeros(size(b3));

y = zeros(pred, bsize);
loss = 0;
for i = 1:bsize
    input = x(:,i);
    target = d(:,i);
    
    y1 = w1*input + b1;
    y2 = relu(y1);
    y3 = w2*y2 + b2;
    y4= relu(y3);
    y5 = w3*y4 + b3;
    y(:, i) = y5;
    
    e = target - y5;
    loss = loss + 0.5*sum(e.^2);
    
    dy5 = -e;
    dw3 = dw3 + dy5*y4';
    db3 = dy5;
    
    dy4 = w3'*dy5;
    dy3 = dy4.*(y3 > 0);
    dw2 = dw2 + dy3*y2';
    db2 = dy3;
    
    dy2 = w2'*dy3;
    dy1 = dy2.*(y1 > 0);
    dw1 = dw1 + dy1*input';
    db1 = db1 + dy1;
end

dw1 = dw1/bsize;    dw2 = dw2/bsize;    dw3 = dw3/bsize;
db1 = db1/bsize;    db2 = db2/bsize;    db3 = db3/bsize;

end
