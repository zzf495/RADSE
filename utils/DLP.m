function [W,M,G]=DLP(Y,E,D,C,u,param,V,W,M)
d2=size(V,1);
G=Y+E+D.*M+C*(1/u);
for i=1:2000
    W=(param.alpha*u*G*V')/(param.alpha*u*(V*V')+2*param.gamma*eye(d2));
    K=W*V-Y-E-C*(1/u);
    MM=K.*D;
    [ss1,ss2]=size(MM);
    for j=1:ss1
        for s=1:ss2
            M(j,s)=max(MM(j,s),0);
        end
    end
    obj(i)=(param.alpha*u/2)*trace((W*V-G)*(W*V-G)')+param.gamma*trace((W*W'));
    if i>2
        if abs(obj(i)-obj(i-1))<0.01
            break
        end
    end
end




