clear all; close all;

% ============ AdaBoost ==============
% loading a given datasets.
% The mat files have been generated so that it contains a structure with 2 members:
%  - X: a n by d matrix that contains the input values
%  - Y: a column vector of n labels 0 or 1
rawData = load('../../resources/datasets/diabetes.mat');
% rawData = load('../../resources/datasets/ionosphere.mat');

% Transformation des 0 en -1.
rawData.Y(rawData.Y == 0) = -1;

% Création du PRDataSet de la bibliothèque PRTools.
data = prdataset(rawData.X, rawData.Y);

% Séparation des données en données d'apprentissage et en données de test.
[dataApp, dataTest] = gendat(data, 0.6);

T = 100; % Nombre de classifieurs que l'on veut apprendre.

% Apprentissage des classifieurs sur les données d'apprentissage.
% Utilisation avec 'tree' ou avec 'stump'
[classifiers, weights] = adaboostLearn(dataApp, T, 'tree');

% Prédiction via l'ensemble de classifieur sur les données de test.
predictions = adaboostPred(classifiers, weights, dataTest);

% Récupération depuis le PRDataSet des vrais label.
trueLabels = getlabels(dataTest);

% Calcul du nombre d'erreurs par rapport aux vraies étiquettes.
testError = sum(predictions ~= trueLabels) / length(predictions) * 100;
fprintf('Error on dataTest for AdaBoost method with binary stamp classifiers: %f\n', testError);
