clear; close all

% ============ AdaBoost ==============
% loading a given datasets.
% The mat files have been generated so that it contains a structure with 2 members:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n labels 0 or 1
% rawData = load('../../resources/datasets/diabetes.mat');
rawData = load('../../resources/datasets/ionosphere.mat');

% creation of prtools dataset
data = prdataset(rawData.X, rawData.Y);
[dataApp, dataTest] = gendat(data, 0.6);

T = 100; % the number of classifiers

% create a new structure and learn
[classifiers, weights] = adaboostLearn(dataApp, T);

predictions = adaboostPred(classifiers, weights, dataTest);
trueLabels = getlabels(dataTest);

testError = length(find(predictions ~= trueLabels)) / length(predictions) * 100;

fprintf('Error on dataTest for AdaBoost method with binary stamp classifiers: %f\n', testError);

clear; close all;
% ============ AdaBoost.M1 ==============
% loading a given datasets.
% The mat files have been generated so that it contains two matrices:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n output values (class indexes)
data = load('../../resources/datasets/synth4.mat');

% split the data
[dataApp.X, dataApp.Y, dataTest.X, dataTest.Y] = splitdata(data.X, data.Y, 0.7);

T = 10; % the number of classifiers

% create a new structure and learn
[classifiers, weights] = adaboostM1Learn(dataApp, T);

predictions = adaboostPred(classifiers, weights, dataTest.X);
testError = length(find(predictions ~= dataTest.Y)) / length(predictions) * 100;

fprintf('Error on dataTest for AdaBoost.M1 method with binary stamp classifiers: %f\n', testError);

% data = load('datasets/synth4.mat');

% T = 10;
% [xApp, yapp, xtest, ytest] = splitdata(data.X, data.Y, 0.7);


% data.X = xapp;
% data.Y = yapp;
% [classifieurs, poids] = adaboostM1(data, T);

% [ntest, p] = size(xtest);
% predictions = zeros(T, ntest);
% for i = 1:T
% 	predictions(i, :) = souchebinaireval(classifieurs{i}, xtest);
% end

% C = 4;
% yTestPreditChaqueClasse = zeros(C, ntest);
% for k = 1:C
% 	predictionsK = predictions == k;
% 	yTestPreditChaqueClasse(k, :) = (predictionsK' * cell2mat(poids)')';
% end

% [val, yTestPredit] = max(yTestPreditChaqueClasse);
% yTestPredit = yTestPredit';

% % [ind, yTestPredit] = max(predictions' * cell2mat(poids)');
% erreurM1 = length(find(yTestPredit ~= ytest)) / ntest * 100