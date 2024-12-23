function ErrorRate = ErrRate(X,Y)
% 归一化
maxV = max(X,[],1);
minV = min(X,[],1);
X = (X - minV)./(maxV-minV);
% K-fold cross-validation
fold = 5;
Indices   =  crossvalind('Kfold', length(Y), fold);
% error rate
ErrRate = zeros(1,fold);
for i = 1 : fold
    train_X = X(Indices~=i,:);
    train_Y = Y(Indices~=i,:);
    test_X = X(Indices==i,:);
    test_Y = Y(Indices==i,:);
    model = fitcknn(train_X,train_Y,'NumNeighbors',5);
    pred_Y = predict(model,test_X);  
    ErrRate(i) = getBalanceError(pred_Y,test_Y);  
end
ErrorRate = mean(ErrRate);
end

function error = getBalanceError(predict,label)
    flag = predict==label;
    
    tbl = tabulate(label);
    labelClass = tbl(:,1);
    classNum = size(tbl,1);
    classAcc = ones(classNum,1);
    for i = 1: classNum
        idx = label == labelClass(i);
        right = sum(flag(idx));
        classAcc(i) = right/tbl(i,2);
    end
    error =1 - mean(classAcc(~isnan(classAcc)));
end