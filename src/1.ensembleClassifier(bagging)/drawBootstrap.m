%% drawBootstrap: function description
function [bag, oob] = drawBootstrap(nTotal, nbTirage)

	bag = randi(nTotal, nbTirage, 1);
	oob = setdiff([1:nTotal], bag)';
end
