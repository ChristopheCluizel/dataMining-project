
clear all; close all;
% ============ AdaBoost.M1 ==============
% loading a given datasets.
% The mat files have been generated so that it contains two matrices:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n output values (class indexes)
% rawData = load('../../resources/datasets/synth4.mat');
rawData = load('../../resources/datasets/synth8.mat');
% rawData = load('../../resources/datasets/segment.mat');

% creation of prtools dataset
data = prdataset(rawData.X, rawData.Y);
[dataApp, dataTest] = gendat(data, 0.6);

T = 600; % the number of classifiers

% create a new structure and learn
[classifiers, weights] = adaboostM1Learn(dataApp, T);

predictions = adaboostM1Pred(classifiers, weights, dataTest);

testError = length(find(predictions ~= getlabels(dataTest))) / length(predictions) * 100;
fprintf('Error on dataTest for AdaBoost.M1 method with binary stamp classifiers: %f\n', testError);