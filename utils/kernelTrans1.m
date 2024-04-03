function [ Ktr1, Ktr2, Kte1, Kte2, Kanchor1, Kanchor2 ] = kernelTrans1( I_tr, T_tr, I_te, T_te, anchorIndex )
%KERNELTRANS 
% in:
%   I_tr, T_tr£ºtraining data£¬N * d
%   I_te, T_te£ºtesting data£¬n * d
%   anchorNum£ºanchor number
% out:
%   Ktr1, Ktr2£ºtraining kernel matrix£¬N * m
%   Kte1, Kte2£ºtesting kernel matrix£¬n * m
%   Kanchor1,2£ºanchor kernel matrix£¬m * m
%   anchorIndex£º
    
    n = size(I_tr, 1);
    
     % random RBF anchor
    anchor1 = I_tr(anchorIndex, :);
    anchor2 = T_tr(anchorIndex, :);

    % ¦Ò^2
    z = I_tr * I_tr';
    z = repmat(diag(z), 1, n)  + repmat(diag(z)', n, 1) - 2 * z;
    sigma1 = mean(z(:));
    clear z;

    z = T_tr * T_tr';
    z = repmat(diag(z), 1, n)  + repmat(diag(z)', n, 1) - 2 * z;
    sigma2 = mean(z(:));
    clear z;

    %
    Kanchor1 = kernelMatrix1(anchor1, anchor1, sigma1);
    Kanchor2 = kernelMatrix1(anchor2, anchor2, sigma2);
    Ktr1 = kernelMatrix1(I_tr, anchor1, sigma1);
    Ktr2 = kernelMatrix1(T_tr, anchor2, sigma2);
    Kte1 = kernelMatrix1(I_te, anchor1, sigma1);
    Kte2 = kernelMatrix1(T_te, anchor2, sigma2);

end

