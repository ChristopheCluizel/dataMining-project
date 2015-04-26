

function proba = rfPredict(X,forest)

	probas = zeros(forest.nbTrees, ); %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	for i = 1:forest.nbTrees
		probas(i) = treePredict(X,forest.trees{i});
	end

	proba = mean(probas);
end
