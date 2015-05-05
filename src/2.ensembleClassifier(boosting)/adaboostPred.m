%% adaboostPred: predict labels
function Y = adaboostPred(classifiers, weights, X)

	[n, p] = size(X);
	T = length(classifiers);

	predictions = zeros(T, n);
	for i = 1:T
		predictions(i, :) = souchebinaireval(classifiers{i}, X);
	end

	Y = sign(predictions' * cell2mat(weights)');
end