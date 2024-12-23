function OffDec = LocalDE(Par1,Par2,Par3,D,Lower,Upper)
% CR = 1 - 0.5*(gen/100);
CR = 0.9;
F = 0.5;
Site = rand(1,D) < CR;
OffDec = Par1;
OffDec(Site) = OffDec(Site) + F.*(Par2(Site)-Par3(Site));
OffDec       = min(max(OffDec,Lower),Upper);
end