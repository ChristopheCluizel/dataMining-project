%% drawBootstrap: function description
function [bag, oob] = drawBootstrap(nTot, nEchan)

	bag = randi(nTot, 1, nEchan);
	oob = setdiff([1:nTot], bag);
end