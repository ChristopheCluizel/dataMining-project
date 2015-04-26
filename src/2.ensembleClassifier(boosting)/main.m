clear; close all

data = load('datasets/diabetes.mat');

data.Y(data.Y == 0) = -1;

T = 1000;
[xapp, yapp, xtest, ytest] = splitdata(data.X, data.Y, 0.7);


data.X = xapp;
data.Y = yapp;
[classifieurs, poids] = adaboost(data, T);

[ntest, p] = size(xtest);
predictions = zeros(T, ntest);
for i = 1:T
	predictions(i, :) = souchebinaireval(classifieurs{i}, xtest);
end


yTestPredit = sign(predictions' * cell2mat(poids)');
erreur = length(find(yTestPredit ~= ytest)) / ntest * 100


%% adaboost.M1

data = load('datasets/synth4.mat');

T = 10;
[xapp, yapp, xtest, ytest] = splitdata(data.X, data.Y, 0.7);


data.X = xapp;
data.Y = yapp;
[classifieurs, poids] = adaboostM1(data, T);

[ntest, p] = size(xtest);
predictions = zeros(T, ntest);
for i = 1:T
	predictions(i, :) = souchebinaireval(classifieurs{i}, xtest);
end

C = 4;
yTestPreditChaqueClasse = zeros(C, ntest);
for k = 1:C
	predictionsK = predictions == k;
	yTestPreditChaqueClasse(k, :) = (predictionsK' * cell2mat(poids)')';
end

[val, yTestPredit] = max(yTestPreditChaqueClasse);
yTestPredit = yTestPredit';

% [ind, yTestPredit] = max(predictions' * cell2mat(poids)');
erreurM1 = length(find(yTestPredit ~= ytest)) / ntest * 100