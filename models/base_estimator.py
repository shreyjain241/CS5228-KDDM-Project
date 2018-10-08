from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from time import time
from datetime import timedelta

def train(X, y, model_name='sgd', model_path='./'):

    if(model_name.lower()  == 'svm'):
        model = SVC(C=1, kernel='rbf', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                    probability=False, tol=0.0001, cache_size=200, class_weight='balanced', verbose=False,
                    max_iter=-1, decision_function_shape='ovr', random_state=None)
        train_start = time()
        model.fit(X, y)
        train_end = time()
        test_start = time()
        pred = model.predict(X)
        test_end = time()
        f1 = metrics.f1_score(y, pred, average='macro')
        print("f1 using SVM is: {} | train time: {} | test time: {}".format(f1, timedelta(seconds=train_end - train_start),
                timedelta(seconds=test_end - test_start)))

    if(model_name.lower()  == 'pa'):
        model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                    shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                    warm_start=False,
                    class_weight='balanced', average=True, n_iter=None)
        train_start = time()
        model.fit(X, y)
        train_end = time()
        test_start = time()
        pred = model.predict(X)
        test_end = time()
        f1 = metrics.f1_score(y, pred, average='macro')
        print("f1 using PA is: {} | train time: {} | test time: {}".format(f1, timedelta(seconds=train_end - train_start),
                timedelta(seconds=test_end - test_start)))

    if(model_name.lower()  == 'sgd'):
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                      max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None,
                      learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                      average=False, n_iter=None)
        train_start = time()
        model.fit(X, y)
        train_end = time()
        test_start = time()
        pred = model.predict(X)
        test_end = time()
        f1 = metrics.f1_score(y, pred, average='macro')
        print("f1 using SGD is: {} | train time: {} | test time: {}".format(f1, timedelta(seconds=train_end - train_start),
                timedelta(seconds=test_end - test_start)))

    if(model_name.lower()  == 'rf'):
        model = RandomForestClassifier(n_estimators=100)
        train_start = time()
        model.fit(X, y)
        train_end = time()
        test_start = time()
        pred = model.predict(X)
        test_end = time()
        f1 = metrics.f1_score(y, pred, average='macro')
        print("f1 using RF is: {} | train time: {} | test time: {}".format(f1, timedelta(seconds=train_end - train_start),
                timedelta(seconds=test_end - test_start)))

    joblib.dump(model, model_path)

