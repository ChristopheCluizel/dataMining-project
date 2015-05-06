clear all; close all;

% ============ AdaBoost ==============
% loading a given datasets.
% The mat files have been generated so that it contains a structure with 2 members:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n labels 0 or 1
rawData = load('../../resources/datasets/diabetes.mat');
rawData.Y(rawData.Y == 0) = -1;
% rawData = load('../../resources/datasets/ionosphere.mat');

% creation of prtools dataset
data = prdataset(rawData.X, rawData.Y);
[dataApp, dataTest] = gendat(data, 0.6);

T = 300; % the number of classifiers

% create a new structure and learn
[classifiers, weights] = adaboostLearn(dataApp, T);

predictions = adaboostPred(classifiers, weights, dataTest);
trueLabels = getlabels(dataTest);

testError = length(find(predictions ~= trueLabels)) / length(predictions) * 100;

fprintf('Error on dataTest for AdaBoost method with binary stamp classifiers: %f\n', testError);
