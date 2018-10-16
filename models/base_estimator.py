from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from time import time
from datetime import timedelta
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

def train(dataset, model_name='sgd'):

    X_train, y_train, X_val, y_val, X_test = dataset

    if(model_name.lower()  in ['svm', 'all']):
        model = SVC(C=1, kernel='rbf', degree=4, gamma='auto', coef0=0.0, shrinking=True,
                    probability=False, tol=0.0001, cache_size=200, class_weight='balanced', verbose=False,
                    max_iter=-1, decision_function_shape='ovr', random_state=None)
        train_start = time()
        model.fit(X_train, y_train)
        train_end = time()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_f1 = metrics.f1_score(y_train, train_pred, average='macro')
        val_f1 = metrics.f1_score(y_val, val_pred, average='macro')
        val_acc = metrics.accuracy_score(y_val, val_pred)
        print("SVM Performance: train F1: {} | train time: {} | val F1: {} | val Acc: {}".
              format(train_f1, timedelta(seconds=train_end - train_start), val_f1, val_acc))

    if(model_name.lower()  in ['pa', 'all']):
        model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=0.001,
                    shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None,
                    warm_start=False,
                    class_weight='balanced', average=True, n_iter=None)
        train_start = time()
        model.fit(X_train, y_train)
        train_end = time()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_f1 = metrics.f1_score(y_train, train_pred, average='macro')
        val_f1 = metrics.f1_score(y_val, val_pred, average='macro')
        val_acc = metrics.accuracy_score(y_val, val_pred)
        print("PA Performance: train F1: {} | train time: {} | val F1: {} | val Acc: {}".
              format(train_f1, timedelta(seconds=train_end - train_start), val_f1, val_acc))

    if(model_name.lower() in ['sgd', 'all']):
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                      max_iter=None, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None,
                      learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                      average=False, n_iter=None)
        train_start = time()
        model.fit(X_train, y_train)
        train_end = time()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_f1 = metrics.f1_score(y_train, train_pred, average='macro')
        val_f1 = metrics.f1_score(y_val, val_pred, average='macro')
        val_acc = metrics.accuracy_score(y_val, val_pred)
        print("SGD Performance: train F1: {} | train time: {} | val F1: {} | val Acc: {}".
              format(train_f1, timedelta(seconds=train_end - train_start), val_f1, val_acc))

    if(model_name.lower() in ['rf', 'all']):
        model = RandomForestClassifier(n_estimators=100)
        train_start = time()
        model.fit(X_train, y_train)
        train_end = time()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_f1 = metrics.f1_score(y_train, train_pred, average='macro')
        val_f1 = metrics.f1_score(y_val, val_pred, average='macro')
        val_acc = metrics.accuracy_score(y_val, val_pred)
        print("RF Performance: train F1: {} | train time: {} | val F1: {} | val Acc: {}".
              format(train_f1, timedelta(seconds=train_end - train_start), val_f1, val_acc))

    if(model_name.lower() in ['ovr', 'all']):
        base_estimator = RandomForestClassifier(n_estimators=100)
        model = OneVsRestClassifier(estimator = base_estimator)
        train_start = time()
        model.fit(X_train, y_train)
        train_end = time()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_f1 = metrics.f1_score(y_train, train_pred, average='macro')
        val_f1 = metrics.f1_score(y_val, val_pred, average='macro')
        val_acc = metrics.accuracy_score(y_val, val_pred)
        print("OvR Performance: train F1: {} | train time: {} | val F1: {} | val Acc: {}".
              format(train_f1, timedelta(seconds=train_end - train_start), val_f1, val_acc))

    return(model)

