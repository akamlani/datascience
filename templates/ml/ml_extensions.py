from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

import statsmodels.api as sm
from sklearn import neighbors


def kfolds_cross_validate(X, y, classifier, k_fold) :
    '''generic cross validation kfolds function'''
    # example usage; cross_validate(np_datavec, np_targetvec, KNeighborsClassifier(k).fit, 5)

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold( len(X), n_folds=k_fold,
                           indices=True, shuffle=True,
                           random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :
        model = classifier(X[ train_slice  ], y[ train_slice  ])
        k_score = model.score(X[ test_slice ], y[ test_slice ])
        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold


# http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
def simple_classification_cv(X, y, test_size_split=0.2):
  '''K nearest neighbors classification using standard 80/20 split'''
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split)
  clf = neighbors.KNeighborsClassifier()
  clf.fit(X_train, y_train)
  clf.score(X_test, y_test)
  return clf


def create_linear_regression_model(X, y):
  X = sm.add_constant(x, prepend=True)
  results = sm.OLS(y, X).fit()
  return results




