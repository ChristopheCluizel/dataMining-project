%% adaboost: function description
function [H, theta] = adaboost(data, T)

	[n, p] = size(data.X);
	poids = ones(n, 1) ./ n;

	for t = 1:T
		H{t} = souchebinairetrain(data.X, data.Y, poids);

		predictions = souchebinaireval(H{t}, data.X);
		epsilon{t} = sum(poids(predictions ~= data.Y));

		theta{t} = 1/2 * log((1 - epsilon{t}) / epsilon{t});

		for i = 1:n
			poids(i) = poids(i) * exp(-theta{t} * data.Y(i) * predictions(i));
		end
		sumPoids = sum(poids);
		poids = poids ./ sumPoids;
	end
end