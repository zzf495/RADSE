function [ WX,WY,B,V] = train_RADSE(Label ,X1, X2, param)
%% Official codes of RADSE
%%%     Authors                      Teng et al.
%%%
%%%     Title                        Robust Asymmetric Cross-Modal Hashing Retrieval 
%%%                                  With Dual Semantic Enhancement
%%%
%% Intput
%%%
%%%     Label                        The label set of samples
%%%
%%%     X1                           The feature of the first modality
%%%
%%%     X2                           The feature of the second modality
%%%
%% param
%%%     nbits                        The length of hash codes
%%%
%%%     lambda1/lambda2              The weight of \| P_k \phi (X^k) - Q_k B \|
%%%
%%%     gamma                        The weight of regularization w.r.t P_k & W
%%%
%%%     alpha                        The weight of RLSLR
%%%
%%%     lambda                       The weight to update V
%%%
%%%     pro                          The increase rate of ADMM
%%%
%%%     u                            The initial weight of ADMM  
%%%
%%%     anchorNum                    The number of anchors
%%%
%% Output
%%%
%%%     WX                      	 Projection of the first modality
%%%
%%%     WY                      	 Projection of the second modality
%%%
%%%     B                            Learned hashing code of samples
%%%
%%% V                            Auxiliary matrix of `B`
%%%
%% Version
%%%
%%%     Upload                   2024-04-03
%%%

L = Label;
nbits = param.nbits;
%% initializaiton
[d1, col] = size(X1);
[d2,~] = size(X2);
n = size(L,2);
c = size(L,1);
sampleColumn = 2*param.nbits;

[v1,v2]=size(L);%n*c
M=rand(v1,v2);
C=ones(v1,v2);
pro= param.pro;
u=param.u;
E=zeros(v1,v2);
% initization B V G
D = L; D(D==0) = -1; %n*c
B = ones(col, nbits);
B(randn(col, nbits) < 0) = -1;
V = ones(col, nbits);
V(randn(col, nbits) < 0) = -1;

G = randn(c,nbits); %c*r
P1 = randn(nbits ,d1); %r*dt
P2 = randn(nbits ,d2);

u0=1e6;
iter = 1;

while (iter<=param.max_iter)

    %update Q1
    [U1,~,Z]=svd(param.lambda1*P1*X1*B,0);
    Q1=U1*Z'; %r*r
    %update Q2
    [U1,~,Z]=svd(param.lambda2*P2*X2*B,0);
    Q2=U1*Z'; %r*r

    %update P1 r*dt
    P1=(param.lambda1*Q1*B'*(X1)')/(param.lambda1*X1*(X1)'+param.gamma*eye(d1));
    P2=(param.lambda2*Q2*B'*(X2)')/(param.lambda2*X2*(X2)'+param.gamma*eye(d2));

    %update M,G
    [G,M,Q]=DLP(L,E,D,C,u,param,B',G,M);
    %G r*c M n*c
    H=G*B'-L-D.*M-C*(1/u);
    E = ALM_E(H,u);
    C=C+u*(E-G*B'+L+D.*M);
    u=min(u0,pro*u);

    % test instance
    Sc = randperm(n,sampleColumn);
    L=L';
    SX = L*L(Sc,:)'>0;
    SY = L(Sc,:)*L'>0;
    V = updateColumnV(V,B,SX,Sc,param,sampleColumn,X1);
    B = updateColumnB(B,V,SY,Sc,param,sampleColumn,X1,X2,P1,P2,Q1,Q2,G,Q,u);
    L=L';
    disp("iter+"+iter);
    iter = iter + 1;


end
WX=(Q1)'*P1;
WY=(Q2)'*P2;

end


