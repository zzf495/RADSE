function [ kM ] = kernelMatrix1( trXA, trXB, parameter )
% calculation of a kernel matrix
% trXA: n1*f feature matrix, n1: the number of instances, f: feature dimensionality
% trXB: n2*f feature matrix, n2: the number of instances, f: feature dimensionality
% kernelObj: kernel settings, RBF kernel here
    trmA = size(trXA, 1);
    trmB = size(trXB, 1);
    tmp0 = trXA * trXB';
    dA = sum(trXA .* trXA, 2);
    dB = sum(trXB .* trXB, 2);
    Cov = repmat(dA, 1, trmB) + repmat(dB', trmA, 1) - 2 * tmp0;
    kM = exp(-(Cov ./ parameter));
end