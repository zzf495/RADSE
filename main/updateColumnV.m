function U = updateColumnV(U,B,S,Sc,param,sampleColum,X1)
m = sampleColum;
n = size(X1,2);
nbits = param.nbits;
lamba = param.lamba;
for k=1:nbits
    TX = param.lamba*U*B(Sc,:)'/nbits;
    AX = 1./(1+exp(-TX));
    Bjk = B(Sc,k)';
    t1=lamba*((S - AX).*repmat(Bjk,n,1))*ones(m,1)/nbits;
    t2=(m*lamba^2)*U(:,k)/(4*nbits^2);
    p=t1+t2;
    U_opt = ones(n,1);
    U_opt(p<0)=-1;
    U(:,k)=U_opt;
end