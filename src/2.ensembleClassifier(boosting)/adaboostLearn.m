%% adaboostLearn: train "T" classifiers on data with AdaBoost method
%%
%% data: prtools dataset to learn.
%% T: the number of classifiers to train.
%% return
%%      classifiers: the set of classifiers for the "tree" implementation.
%%      theta: weights for each classifier.

function [classifiers, theta] = adaboostLearn(data, T)

	[n, p] = size(data);

	% intialize weights
	weight = ones(n, 1) ./ n;

	for t = 1:T
		fprintf('Compute classifier %d\n', t);
		weightData = gendatw(data, weight, 100 * n);

		classifiers{t} = stumpc(weightData);
		predictions = labeld(data, classifiers{t});
		trueLabels = getlabels(data);

		epsilon{t} = sum(weight(predictions ~= trueLabels));

		theta{t} = 1/2 * log((1 - epsilon{t}) / epsilon{t});

		% compute new weights
		for i = 1:n
			if trueLabels(i) == predictions(i)
				weight(i) = weight(i) * exp(-theta{t});
			else
				weight(i) = weight(i) * exp(theta{t});
			end
		end
		sumWeight = sum(weight);
		weight = weight ./ sumWeight;
	end
end