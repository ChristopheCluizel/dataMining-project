%% adaboost: function description
function [H, theta] = adaboostM1(data, T)

	[n, p] = size(data.X);
	C = length(unique(data.Y));
	poids = ones(n, 1) ./ n;

	for t = 1:T
		H{t} = souchebinairetrain(data.X, data.Y, poids);

		predictions = souchebinaireval(H{t}, data.X);
		epsilon{t} = sum(poids(predictions ~= data.Y));

		theta{t} = log((C - 1) * (1 - epsilon{t}) / epsilon{t});

		for i = 1:n
			poids(i) = poids(i) * theta{t} * (predictions(i) ~= data.Y(i));
		end
		sumPoids = sum(poids);
		poids = poids ./ sumPoids;
	end
end