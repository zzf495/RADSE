close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./main/'));
rng('default');
result_URL = './results/';
if ~isfolder(result_URL)
    mkdir(result_URL);
end
db = {'NUSWIDE'};
hashmethods = {'RADSE'};
loopnbits = [8 16 32 64 96 128]; 
RADSEparam.lambda1=0.01;
RADSEparam.lambda2=0.01;
RADSEparam.lamba=8;
RADSEparam.alpha=0.1;
RADSEparam.max_iter=15;
RADSEparam.gamma=1e-5;
RADSEparam.u=1;
RADSEparam.pro=1.01;
RADSEparam.anchorNum=1300;
for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    diary(['./results/res_',db_name,'_result.txt']);
    diary on;

    %% load dataset
    load(['./datasets/',db_name,'.mat']);
    result_name = [result_URL 'RADSE_' db_name '_result' '.mat'];
    XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
    XTest = I_te; YTest = T_te; LTest = L_te;
    clear X Y L
    clear I_tr I_te L_tr L_te

    %% Label Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end

    %% Methods
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        RADSEparam.dataname = db_name;
        RADSEparam.nbits = loopnbits(ii);
        for jj = 1:length(hashmethods)
            eva_info_ = evaluate_RADSE(XTrain,YTrain,LTrain,XTest,YTest,LTest,RADSEparam);
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    %% Results
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
        end
        fprintf("%dbits  I2T = %f ; T2I = %f ;  \n",loopnbits(ii),Image_VS_Text_MAP{jj,ii},Text_VS_Image_MAP{jj,ii});
    end
    % I -> T
    % T -> I
    % Training time
    % Query time
    save(result_name,'Image_VS_Text_MAP','Text_VS_Image_MAP','trainT');
    diary off;
end