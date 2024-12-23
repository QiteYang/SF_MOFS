function [index,KN_point] = KneeSelect(PopObj)

[~,index] = sort(PopObj(:,1),'ascend');
axis_x = PopObj(index,1);
axis_y = PopObj(index,2);
point = [axis_x';axis_y'];

N = size(PopObj,1);
dis = zeros(1,N);
for i = 1:size(PopObj,1)
    dis(i) = abs(det([point(:,N)-point(:,1),point(:,i)-point(:,1)]))/norm(point(:,N)-point(:,1));
end
[~,KN_point] = max(dis);

end

