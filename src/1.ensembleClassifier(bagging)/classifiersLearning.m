%% classifiersLearning: train "K" classifiers on X_app
%%
%% X_app: the data to learn with.
%% K: the number of classifiers to train.
%% return the set of classifiers for the "tree" implementation.

function classifiers = classifiersLearning(X_app, K)
    [m, n] = size(X_app);
    for i = 1:K
        [bag, oob] = drawBootstrap(m, m);
        Dk = X_app(bag, :);
        % training a decision tree classifier on X_app
        classifiers{i} = treec(Dk);
    end
end
