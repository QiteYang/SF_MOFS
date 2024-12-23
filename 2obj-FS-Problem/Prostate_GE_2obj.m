classdef Prostate_GE_2obj < PROBLEM
    % <multi> <real> <large/none> <expensive/none>
    
    properties(Access = private)
        X_train;	% Profit of each item
        Y_train;  % Weight of each item
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            load Prostate_GE.mat;
            Pro = 0.7; % 训练数据比例
            len_X = size(X,1);
            X_train = X(1:floor(Pro*len_X),:);
            Y_train = Y(1:floor(Pro*len_X),:);
            obj.M = 2;
            obj.D = 5966;
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.X_train = X_train;
            obj.Y_train = Y_train;
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            X_train = obj.X_train;
            Y_train = obj.Y_train;
            [N,D] = size(PopDec);
            PopObj = zeros(N,2);
            for i = 1 : N
                dec = PopDec(i,:);
                Feature = find(dec >= 0.6);
                if isempty(Feature)
                    PopObj(i,1) = 1;
                    PopObj(i,2) = 1;
                else
                    newdata_X = X_train(:,Feature);
                    PopObj(i,1) = ErrRate(newdata_X,Y_train); % error rate
                    PopObj(i,2) = length(Feature)./D;
                end
            end
        end
    end
end