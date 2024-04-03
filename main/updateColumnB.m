function B = updateColumnB(B,U,S,Sc,param,sampleColumn,X1,X2,P1,P2,Q1,Q2,G,Q,u)
m=sampleColumn;
n=size(U,1);
bits = param.nbits;
lamba = param.lamba;
alpha = param.alpha;
lambda1 = param.lambda1;
lambda2 = param.lambda2;
for k=1:bits
    TX1 = lamba * U(Sc,:) * B' /bits;
    AX1 = 1 ./ (1 + exp(-TX1));
    Ujk = U(Sc,k)';
    aa=(alpha*u/2)+lambda1+lambda2;
    t1=param.lamba* ((S' - AX1') .* repmat(Ujk, n, 1)) * ones(m, 1)  / bits;
    t2=(m*lamba^2+8*aa*bits^2)*B(:,k)/(4*bits^2);
    temp3 = B(:,k)*(G(:,k)')*G(:,k)-Q'*G(:,k);
    temp5 = B(:,k)*(Q1(:,k))'*Q1(:,k)-(X1)'*P1(k,:)'*Q1(k,k);
    temp6 = B(:,k)*(Q2(:,k))'*Q2(:,k)-(X2)'*P2(k,:)'*Q2(k,k);
    t3 = -alpha*u*temp3 - 2 *lambda1*temp5 - 2*lambda2*temp6;
    p=t1+t2-t3;
    B_opt=ones(n,1);
    B_opt(p < 0) = -1;
    B(:,k) = B_opt;
end
end