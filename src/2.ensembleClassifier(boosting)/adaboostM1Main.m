
clear all; close all;
% ============ AdaBoost.M1 ==============
% loading a given datasets.
% The mat files have been generated so that it contains two matrices:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n output values (class indexes)
% rawData = load('../../resources/datasets/synth4.mat');
rawData = load('../../resources/datasets/synth8.mat');
% rawData = load('../../resources/datasets/segment.mat');

% Création du PRDataSet de la bibliothèque PRTools.
data = prdataset(rawData.X, rawData.Y);

% Séparation des données en données d'apprentissage et en données de test.
[dataApp, dataTest] = gendat(data, 0.6);

T = 20; % Nombre de classifieurs que l'on veut apprendre.

% Apprentissage des classifieurs sur les données d'apprentissage.
[classifiers, weights] = adaboostM1Learn(dataApp, T, 'tree');

% Prédiction via l'ensemble de classifieur sur les données de test.
predictions = adaboostM1Pred(classifiers, weights, dataTest);

% Calcul du nombre d'erreurs par rapport aux vraies étiquettes.
testError = sum(predictions ~= getlabels(dataTest)) / length(predictions) * 100;
fprintf('Error on dataTest for AdaBoost.M1 method with binary stamp classifiers: %f\n', testError);