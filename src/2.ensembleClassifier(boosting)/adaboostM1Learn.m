%% adaboostM1Learn: train "T" classifiers on data with AdaBoost.M1 method
%%
%% data: the data to learn with (X and Y).
%% T: the number of classifiers to train.
%% return
%%      classifiers: the set of classifiers for the "tree" implementation.
%%      theta: weights for each classifier.

function [classifiers, theta] = adaboostM1Learn(data, T)

	[n, p] = size(data.X);
	C = length(unique(data.Y));
	weights = ones(n, 1) ./ n;

	for t = 1:T
		classifiers{t} = souchebinairetrain(data.X, data.Y, weights);

		predictions = souchebinaireval(classifiers{t}, data.X);
		epsilon{t} = sum(weights(predictions ~= data.Y));

		theta{t} = log((C - 1) * (1 - epsilon{t}) / epsilon{t});

		for i = 1:n
			weights(i) = weights(i) * theta{t} * (predictions(i) ~= data.Y(i));
		end
		sumWeights = sum(weights);
		weights = weights ./ sumWeights;
	end
end