%% drawBootstrap: function description
function [bag, oob] = drawBootstrap(nTotal, nbDraw)

	bag = randi(nTotal, nbDraw, 1);
	oob = setdiff([1:nTotal], bag)';
end
