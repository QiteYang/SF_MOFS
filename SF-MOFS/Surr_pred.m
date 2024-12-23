function offdec = Surr_pred(offdec,model,Problem,Pro_select)
off = offdec;
off(off>=0.6) = 1;
off(off<0.6) = 0;
off_orig = off;
% surrogate prediction for 子代个体
pred_C = 0;
pred_C = svmpredict(pred_C,off,model,'-q');
chosen = find(off == 1);
unchosen = find(off == 0);
if pred_C == 2
    % 预测为二类时，对1的维度flip，直到找到预测为一类的或者超过循环次数
    for j = 1 : 5*Problem.N
        off_flip = off;
        for k = 1 : length(chosen)
            if rand(1) < (1-Pro_select(chosen(k)))*(length(chosen)/Problem.D)
                off_flip(chosen(k)) = 0;
            end
        end
        pred_C = svmpredict(pred_C,off_flip,model,'-q');
        if pred_C == 1
            off = off_flip;
            break;
        elseif pred_C == 2
            off = off_flip;
            chosen = find(off == 1);
            unchosen = find(off == 0);
        end
    end
elseif pred_C == 3
    % 预测为三类的，对所有维度flip
    flip_all = 1;
    for j = 1 : 5*Problem.N
        off_flip = off;
        for k = 1 : length(chosen)
            if rand(1) < (1-Pro_select(chosen(k)))*(length(chosen)/Problem.D)
                off_flip(chosen(k)) = 0;
            end
        end
        if flip_all == 1
            for k = 1 : length(unchosen)
                if rand(1) < Pro_select(unchosen(k))*(length(unchosen)/Problem.D)
                    off_flip(unchosen(k)) = 1;
                end
            end
        end
        pred_C = svmpredict(pred_C,off_flip,model,'-q');
        if pred_C == 1 % flip后预测为一类，直接跳出
            off = off_flip;
            break;
        elseif pred_C == 2 % flip后预测为二类，替换off，但是继续flip
            off = off_flip;
            chosen = find(off == 1);
            unchosen = find(off == 0);
            flip_all = 0;
        end
    end
end
replace = find(off~=off_orig);
offdec(replace) = 1 - offdec(replace);
end