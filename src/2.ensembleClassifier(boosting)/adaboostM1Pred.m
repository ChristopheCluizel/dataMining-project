%% adaboostPred: predict labels
function Y = adaboostPred(classifiers, weights, data)

	[n, p] = size(data);
	T = length(classifiers);
	C = length(unique(data.nlab));

	labelPredictions = zeros(T, n);
	for i = 1:T
		labelPredictions(i, :) = labeld(data, classifiers{i});
	end

	predictionWeightForEachClass = zeros(C, n);
	weightsMatrix = repmat(cell2mat(weights), [n, 1])';
	for k = 1:C
		booleanPredictions{k} = labelPredictions == k;
		predictionWeightForEachClass(k, :) = sum(booleanPredictions{k} .* weightsMatrix);
	end

	[val, Y] = max(predictionWeightForEachClass);
	Y = Y';
end


