#encoding=utf-8

import argparse
import os
import sys
sys.path.append("..")
from utils import data_util
from models import base_estimator
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='Article Classification')
    parser.add_argument('--model', default="svm", type=str,
                        help='one of `svm` or `pa` or `adaboost` or blah blah')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return (args)

if __name__ == "__main__":
    args = get_args()
    proj_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(proj_path, 'data/')
    results_path = os.path.join(proj_path, 'results/')
    model_path = os.path.join(proj_path, 'models/')

    # save model
    model_ckpt_path = os.path.join(model_path, args.model.lower() + '_ckpt.tar')

    # save model history: loss, acc etc
    history_path = os.path.join(results_path, args.model.lower() + '_history.pkl')

    # save test output
    test_output_path = os.path.join(results_path, args.model.lower() + '_test_submission.csv')

    # load data
    train_data, train_labels, test_data = data_util.load_and_process(train_path=os.path.join(data_path, 'train.csv'),
                                                                     test_path=os.path.join(data_path, 'test.csv'))

    # train quick SVM model
    print('training {} estimator...'.format(args.model))
    base_estimator.train(train_data, train_labels, model_path=model_ckpt_path, model_name=args.model)

    # evualuate model
    best_model = joblib.load(model_ckpt_path)
    predicted = best_model.predict(test_data)

    # save results
    output = pd.DataFrame({'article_id': pd.Series(np.arange(1, test_data.shape[0]+1)), 'category': pd.Series(predicted)})
    output.to_csv(test_output_path, sep=',', header=True, index=False)
