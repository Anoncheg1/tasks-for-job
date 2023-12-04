import pandas as pd
from sklearn import datasets
import numpy as np
d = datasets.load_iris()
X = d['data']
y = d['target']
print(X[0:5])
print("y", y)
# -- one hot encoding for y
Y = np.zeros((y.size, y.max()+1))
Y[np.arange(y.size),y] = 1


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.metrics import hinge_loss
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
import sklearn

classifiers_binary = [
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),  # очень долго
        SVC(gamma=2, C=1),  # слишком долго
        GaussianProcessClassifier(1.0 * RBF(1.0)), # не хватает памяти
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, ),  # max_features=1
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
]

def _select_metrics(est1, est2, X, Y, kfold):
    print( '{:40} {:5} {:5}'.format("metric", "dummy model", "one of effective model" ))
    print( '{:40} {:5} {:5}'.format("metric", "mean_accuracy", "std" ))
    for k in metrics.get_scorer_names():
        # print(k)
        results1 = cross_validate(est1, X, Y, cv=kfold, scoring=[k])
        r1 = results1[f'test_{k}']
        results2 = cross_validate(est2, X, Y, cv=kfold, scoring=[k])
        r2 = results2[f'test_{k}']
        if not all(np.isnan(r1)):
            print( '{:40} {:5} {:5}\t{:5} {:5}'.format(k, round(r1.mean(), 3), round(r1.std(),2), round(r2.mean(), 3), round(r2.std(),2)) )

def _check_model_binary(est, X, Y, kfold):
    results = cross_validate(est, X, Y, cv=kfold, scoring=['accuracy', 'roc_auc'])
    print(est.__class__.__name__)
    print("Accuracy: %f" % results['test_accuracy'].mean())
    print("AUC: %f" % results['test_roc_auc'].mean())

def _check_model_multiclass_native(est, X, Y, kfold):
    """ https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    returns score per folds"""
    print(est)
    results = cross_validate(est, X, Y, cv=kfold,
                             scoring=['roc_auc_ovo', 'precision_macro',
                                      'recall_macro'])
    print("ROC_AUC: %f" % results['test_roc_auc_ovo'].mean())
    print("precision_macro: %f" % results['test_precision_macro'].mean())
    print("recall_macro: %f" % results['test_recall_macro'].mean())
    print()

def _check_model_multiclass_ovo(est, X, Y, kfold):
    """ https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    returns score per folds"""
    scoring=['accuracy', 'precision_macro',
                                      'recall_macro']
    results = cross_validate(est, X, Y, cv=kfold, scoring=scoring) #
    print(est)
    for x in scoring:
        print(x+ ": %f" % results['test_'+x].mean())
    print()


classifiers_multiclass_nativ = [
    sklearn.naive_bayes.BernoulliNB(),
    sklearn.tree.DecisionTreeClassifier(),
    sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, ),
    sklearn.tree.ExtraTreeClassifier(),
    sklearn.ensemble.ExtraTreesClassifier(),
    sklearn.naive_bayes.GaussianNB(),
    sklearn.neighbors.KNeighborsClassifier(),
    sklearn.linear_model.LogisticRegression(multi_class="multinomial"),
    sklearn.linear_model.LogisticRegressionCV(multi_class="multinomial")
    ]

classifiers_multiclass_ovo = [
    OneVsOneClassifier(sklearn.svm.LinearSVC(C=100.)),
    OneVsOneClassifier(sklearn.svm.SVC(kernel="linear", C=0.025)),  # очень долго
    OneVsOneClassifier(sklearn.svm.SVC(gamma=2, C=1)),  # слишком долго
    OneVsOneClassifier(sklearn.gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0))), # не хватает памяти
    OneVsOneClassifier(sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000)),
    OneVsOneClassifier(sklearn.ensemble.AdaBoostClassifier()),
    OneVsOneClassifier(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()),
    OneVsOneClassifier(sklearn.ensemble.GradientBoostingClassifier()),
    OneVsOneClassifier(sklearn.gaussian_process.GaussianProcessClassifier()),
    OneVsOneClassifier(sklearn.linear_model.LogisticRegression(multi_class="ovr")),
    OneVsOneClassifier(sklearn.linear_model.LogisticRegressionCV(multi_class="ovr")),
    OneVsOneClassifier(sklearn.linear_model.SGDClassifier()),
    OneVsOneClassifier(sklearn.linear_model.Perceptron())
    ]


kfold = StratifiedKFold(n_splits=5)
# ----------- select metrics 1) dummy ----
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X, y)
dummy_clf.predict(X)
print("predict", dummy_clf.predict(X))
print("score", dummy_clf.score(X, y))
print('---')
# _select_metrics(dummy_clf, X, y, kfold)

# ----------- select metrics 2) model ------
# m = linear_model.LogisticRegressionCV(max_iter=10, multi_class='multinomial')
# m = linear_model.Lasso()
# m=KNeighborsClassifier(5)
m = OneVsOneClassifier(sklearn.ensemble.AdaBoostClassifier())
_select_metrics(dummy_clf, m, X, y, kfold)

# ------------------ select model -----------
# for est in classifiers_multiclass_nativ: # classifiers_multiclass_ovo:
#     _check_model_multiclass_native(est, X, y, kfold)

# for est in classifiers_multiclass_ovo: # classifiers_multiclass_ovo:
#     _check_model_multiclass_ovo(est, X, y, kfold)
