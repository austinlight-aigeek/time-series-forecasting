function [numBatch, lastBSize] = calLastBSize(trainSize, pred, bsize, m)

i = 1;
j = 1;
while(1)
    flag = 0;
    for j = 1:bsize
        ey = (i-1)*bsize*pred + j*pred+m;
        if ey>=trainSize
            flag = 1;
            break;
        end
    end
    if flag==1
        break;
    end
	i = i + 1;
end

numBatch = i;
lastBSize = j;
end