%% Predict the labels on data with a random forest
%% X: the data set
%% forest: the random forest learned
%% return the labels for each data

function probas = rfPredict(X, forest)

	probas = zeros(length(X), 2);

	for i = 1:forest.nbTrees
		probas = probas + treePredict(X,forest.trees{i});
	end

	probas = probas / forest.nbTrees;
end
