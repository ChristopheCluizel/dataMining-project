%% adaboostPred: predict labels
function Y = adaboostPred(classifiers, weights, data)

	[n, p] = size(data);
	T = length(classifiers);

	predictions = zeros(T, n);
	for i = 1:T
		predictions(i, :) = labeld(data, classifiers{i});
	end

	Y = sign(predictions' * cell2mat(weights)');
end