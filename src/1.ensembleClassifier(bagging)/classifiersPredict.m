%% classifiersPredict: predict on "X_test" with a set of classifiers.
%% X_test: the test set.
%% classifiers: the set of classifiers
%% return the predictions for the set of classifiers.

function predictions = classifiersPredict(X_test, classifiers)

    [m, n] = size(X_test);
    K = size(classifiers, 2);

    label = zeros(m, K);
    for i = 1:K
        label(:, i) = labeld(X_test, classifiers{i});
    end

    predictions = zeros(m, 1);
    for j = 1:m
        [nbOccurence, labelOcc] = hist(label(j, :), unique(label(j, :)));
        [val, ind] = max(nbOccurence);
        predictions(j, 1) = labelOcc(ind);
    end
end
