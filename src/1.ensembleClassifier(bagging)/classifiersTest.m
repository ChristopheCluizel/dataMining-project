%% classifiersTest: compute the prediction error.
%% X_test: the test set.
%% classifiers: the set of classifiers.
%% return the error between the predictions and true labels.

function error = classifiersTest(X_test, classifiers)
    [m, n] = size(X_test);

    predictions = classifiersPredict(X_test, classifiers);
    error = 1 - (sum(predictions == X_test.nlab) / m);
end
