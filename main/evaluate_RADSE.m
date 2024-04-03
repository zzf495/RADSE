function evaluation_info=evaluate_RADSE(XTrain,YTrain,LTrain,XTest,YTest,LTest,param)
%% Official codes of RADSE
%%%     Authors                      Teng et al.
%%%
%%%     Title                        Robust Asymmetric Cross-Modal Hashing Retrieval 
%%%                                  With Dual Semantic Enhancement
%%%
%% Intput
%%%
%%%     XTrain/YTrain                The features of the first/second modality
%%%
%%%     LTrain                       The label set of `XTrain` and `YTrain`
%%%
%%%     XTest/YTest                  The features of the first/second modality
%%%                              (test sets)
%%%
%%%     LTest                        The ground-truth of test sets
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
%%%     evaluation_info              Results of RADSE
%%%
%% Version
%%%
%%%     Upload                      2024-04-03
%%%
%% ----------------- initialization ----------------------
n = size(XTrain, 1);
n_anchors =param.anchorNum;
anchor_image =XTrain(randsample(n,n_anchors),:);
anchor_text = YTrain(randsample(n,n_anchors),:);
XKTrain = RBF_fast(XTrain',anchor_image');
XKTest = RBF_fast(XTest',anchor_image');
YKTrain = RBF_fast(YTrain',anchor_text');
YKTest = RBF_fast(YTest',anchor_text');
tic;
%% ----------------- hash learning ----------------------
[ WX,WY,B,V] = train_RADSE( LTrain', XKTrain', YKTrain',param);
HI_te = sign(WX * XKTest');   
HT_te = sign(WY * YKTest');  
%% ----------------- evaluate ----------------- 
traintime=toc;
evaluation_info.trainT=traintime;
tic;
sim_ti = V*HI_te;
R = size(B,1);
evaluation_info.Image_VS_Text_MAP = mAP(sim_ti,LTrain, LTest,R);
sim_it = V*HT_te;
evaluation_info.Text_VS_Image_MAP = mAP(sim_it,LTrain, LTest,R);
compressiontime=toc;
evaluation_info.compressT=compressiontime;
end