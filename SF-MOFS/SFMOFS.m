classdef SFMOFS < ALGORITHM
    % <multi> <real/binary> <constrained/none>
    
    %------------------------------- Reference --------------------------------
    % Q. -T. Yang et al., "Surrogate-Assisted Flip for Evolutionary High-Dimensional
    % Multiobjective Feature Selection," 2024 IEEE Congress on Evolutionary Computation (CEC),
    % Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10612094.
    
    methods
        function main(Algorithm,Problem)
            %% parameter setting
            N = Problem.N;
            D = Problem.D;
            
            % load dataset
            load hillvally.mat;
            Pro = 0.7; % 训练数据比例
            len_X = size(X,1);
            X_train = X(1:floor(Pro*len_X),:);
            Y_train = Y(1:floor(Pro*len_X),:);
            
            %% step 1: moving irrelevant features
            % C-relevance SU
            su_c = zeros(1,D);
            for i = 1 : D
                su_c(i) = MItest(X_train(:,i),Y_train);
                %                 su_c(i) = (SU(X_train(:,i),Y_train,10));
            end
            [su_c_rank,~] = sort(su_c,'ascend');
            p0 = min(0.1*max(su_c), su_c_rank(ceil(D/log(D))));
            Strong = find(su_c >= p0);
            
            %% step 2: FCFC clustering
            dtmax = 0;
            for i = 1 : D
                for j = [1:i-1,i+1:D]
                    if abs(su_c(i) - su_c(j)) > dtmax
                        dtmax = abs(su_c(i) - su_c(j));
                    end
                end
            end
            p1 = dtmax*log(length(Strong))/length(Strong); % rho参数，F-redundancy中使用
            su_temp = su_c(Strong);
            [su_temp,loc] = sort(su_temp,'descend');
            Strong = Strong(loc);
            
            U0 = Strong;
            K = 1;
            Group = {};
            while length(U0) > 1
                U1 = U0;
                temp = [];
                leave = [];
                for i = 2 : length(U1)
                    difference = abs(su_temp(i) - su_temp(1));
                    if difference > p1
                        leave = [leave,i];
                    end
                end
                U1(leave) = [];
                temp = [temp, su_temp(leave)];
                if length(U1) > 1
                    leave = [];
                    for i = 2 : length(U1)
                        su_f = MItest(X_train(:,U1(1)),X_train(:,U1(i)));
                        if su_f < min(su_temp(1), su_temp(i))
                            leave = [leave,i];
                        end
                    end
                    U1(leave) = [];
                end
                temp = [temp, su_temp(leave)];
                Group{K} = U1;
                U0 = setdiff(U0,U1,'stable');
                su_temp = temp;
                K = K + 1;
            end
            K = K - 1;
            
            %% 初始化
            Upper = Problem.upper;
            Lower = Problem.lower;
            Population = Problem.Initialization();
            [FrontNo,~] = NDSort(Population.objs,inf);
            
            %% Optimization
            gen = 0;
            Output = Population(FrontNo==1);
            while Algorithm.NotTerminated(Output)
                %% optimization process
                PopDec = Population.decs;
                PopObj = Population.objs;

                % 计算skilled group
                GR = zeros(N,K);
                for i = 1 : K
                    group = Group{i};
                    decs = PopDec(:,group);
                    decs(decs>=0.6) = 1;
                    decs(decs<0.6) = 0;
                    GR(:,i) = sum(decs,2)./length(group);
                end
                SG = zeros(N,K);
                for i = 1 : N
                    index = find(GR(i,:) == min(GR(i,:)));
                    SG(i,1:length(index)) = index;
                end
                
                % DE for each group
                OffDec = zeros(N,D);
                for i = 1 : K
                    index = [];
                    for j = 1 : N
                        if ~isempty(find(SG(j,:)==i, 1))
                            index = [index,j];
                        end
                    end
                    subP = Population(index);
                    subObj = subP.objs;
                    group = Group{i};
                    % group optima
                    if isempty(subP)
                        [~,loc] = min(PopObj(:,1));
                        Gbest = Population(loc);
                    else
                        [~,loc] = min(subObj(:,1));
                        Gbest = subP(loc);
                    end
                    par1 = Gbest.dec;
                    par1 = par1(group);
                    
                    for j = 1 : N
                        offdec = OffDec(j,group);
                        dec = PopDec(j,group);
                        if isempty(subP)
                            matingpool = TournamentSelection(2,1,FrontNo,PopObj(:,1));
                            par2 = Population(matingpool).dec;
                        else
                            matingpool = TournamentSelection(2,1,subObj(:,1));
                            par2 = subP(matingpool).dec;
                        end
                        par2 = par2(group);
                        offdec = LocalDE(dec,par1,par2,length(group),Lower(group),Upper(group));
                        OffDec(j,group) = offdec;
                    end
                end    
                
                %% surrogate-assisted optimization
                if mod(gen,10) == 0  % 每10代更新一次SVM surrogate     
                    DB = Population;
                    DBObj = DB.objs;
                    [front,~] = NDSort(DBObj,inf);
                    F1 = DB(front==1);
                    if max(front) == 1
                        % knee点划分
                        [index,KN_point] = KneeSelect(DBObj);
                        C_1 = DB(index(1:KN_point));
                        C_2 = [];
                        C_3 = DB(index(KN_point+1:end));
                    else
                        C_1 = F1;
                        OptObj = F1.objs;
                        K_err = max(OptObj(:,1));
                        Fn = DB(front~=1);
                        FnObj = Fn.objs;
                        C_2 = Fn(FnObj(:,1)<=K_err);
                        C_3 = Fn(FnObj(:,1)>K_err);
                    end
                    % 构建DB
                    len_1 = length(C_1);
                    len_2 = length(C_2);
                    len_3 = length(C_3);
                    C = [C_1,C_2,C_3];
                    DB_x = C.decs;
                    % 二进制化
                    DB_x(DB_x>=0.6) = 1;
                    DB_x(DB_x<0.6) = 0;
                    DB_y = [ones(len_1,1);2*ones(len_2,1);3*ones(len_3,1)];
                    model = svmtrain(DB_y,DB_x,'-t 0 -q');
                    DB = []; % model构建完之后重置DB
                    
                    BinDec = zeros(N,D);
                    BinDec(PopDec>=0.6) = 1;
                    BinDec(PopDec<0.6) = 0;
                    Pro_select = sum(BinDec,1)./N;
                    for i = 1 : N
                        OffDec(i,:) = Surr_pred(OffDec(i,:),model,Problem,Pro_select);
                    end
                end

                Offspring = Problem.Evaluation(OffDec);
                
                % Environmental Selection
                [Population,FrontNo,~] = EnvironmentalSelection([Population,Offspring],Problem.N);
                Output = Population(FrontNo==1);
                gen  = gen + 1;
            end
        end
    end
end