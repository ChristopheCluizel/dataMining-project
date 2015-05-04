clear; close all

X = load('skew.mat');
X = X.values';

[n, m] = size(X);
K = 20; % the number of classifiers

% ========= Statistics moment ==========
gamma = [];
for i = 1:K
	[bag, oob] = drawBootstrap(n, n);
	gamma(i) = skewness(X(bag));
end
gamma = gamma';

gammaMeanBootstrap = mean(gamma);
gammaStdBootstrap = std(gamma);
gammaX = skewness(X);

fprintf('Gamma mean with Bootstrap: %0.2f\n', gammaMeanBootstrap);
fprintf('Gamma std with Bootstrap: %0.2f\n', gammaStdBootstrap);
fprintf('Gamma X: %02f\n', gammaX);


% ============ Bagging ==============
% loading a given datasets.
% The mat files have been generated so that it contains two matrices:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n output values (class indexes)
synth4 = load('../../resources/datasets/synth4.mat');

% data are converted into Dataset object from the PRTools library
data = prdataset(synth4.X, synth4.Y);

% stratified random splitting of the dataset
[X, X_test] = gendat(data, 0.6);
[mDs, nDs] = size(X_test);

K = 10; % the number of classifiers
treeClassifiers = classifiersLearning(X, K);

% estimating the test error rates for the trained classifiers
errTree = testc(X_test, treeClassifiers);
fprintf('Mean error for the sets of tree classifiers: %f\n', mean(cell2mat(errTree)))
fprintf('Error for Baggin method with tree classifiers: %f\n', classifiersTest(X_test, treeClassifiers))
