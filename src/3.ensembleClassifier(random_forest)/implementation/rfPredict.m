function probas = rfPredict(X,forest)

	probas = zeros(length(X), 2);

	for i = 1:forest.nbTrees
		probas = probas + treePredict(X,forest.trees{i});
	end

	probas = probas / forest.nbTrees;
end
