%% adaboostLearn: train "T" classifiers on data
%%
%% data: the data to learn with (X and Y).
%% T: the number of classifiers to train.
%% return
%%      classifiers: the set of classifiers for the "tree" implementation.
%%      theta: weights for each classifier.

function [classifiers, theta] = adaboostLearn(data, T)

	[n, p] = size(data.X);

	% intialize weights
	weight = ones(n, 1) ./ n;

	for t = 1:T
		classifiers{t} = souchebinairetrain(data.X, data.Y, weight);

		predictions = souchebinaireval(classifiers{t}, data.X);
		epsilon{t} = sum(weight(predictions ~= data.Y));

		theta{t} = 1/2 * log((1 - epsilon{t}) / epsilon{t});

		% compute new weights
		for i = 1:n
			weight(i) = weight(i) * exp(-theta{t} * data.Y(i) * predictions(i));
		end
		sumWeight = sum(weight);
		weight = weight ./ sumWeight;
	end
end