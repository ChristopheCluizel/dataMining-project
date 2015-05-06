%% adaboostPred: predict labels
function Y = adaboostPred(classifiers, weights, X)

	[n, p] = size(X);
	predictions = zeros(T, n);
	for i = 1:T
		predictions(i, :) = souchebinaireval(classifieurs{i}, X);
	end

	C = 4;
	yTestPreditChaqueClasse = zeros(C, n);
	for k = 1:C
		predictionsK = predictions == k;
		yTestPreditChaqueClasse(k, :) = (predictionsK' * cell2mat(poids)')';
	end

	[val, yTestPredit] = max(yTestPreditChaqueClasse);
	yTestPredit = yTestPredit';
end


