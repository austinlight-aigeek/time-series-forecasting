function [RMSE, MAPE] = eval_error(y, yhat)
    n = length(y);
    % Root Mean Error
    RMSE = sqrt(mean((y-yhat).^2));
    % Mean Absolute Percentage Error
    MAPE = sum(abs(y-yhat)./y)/n*100;
end
