function [Population,FrontNo] = Select(Population,N)
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front based on their crowding distances
    PopObj = Population.objs;
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(PopObj(Last,1),'ascend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
end