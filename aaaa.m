clc
bsize = 16;
m = 30;
pred = 5;
T = 3;

j = 48;

sx = T*pred*(j-1)+1;
ex = sx+m-1;
sy = ex+1;
ey = sy+pred-1;

[sx ex]
[sy ey]
