
import nltk
import string
import csv

import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import scipy.sparse as sp
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import os


print ("Importing data.....")
raw_train_data = pd.read_csv('data' + os.sep + 'train_v2.csv')
raw_test_data = pd.read_csv('data' + os.sep + 'test_v2.csv')
y_train_final = raw_train_data['category'].as_matrix()



train_data = raw_train_data[['article_id','title','publisher','timestamp']]
test_data = raw_test_data[['article_id','title','publisher','timestamp']]



#get categorical features
print ("Getting categorical features.....")
X_all = pd.concat([train_data.copy(),test_data.copy()], ignore_index=True)
publisher_encoder = LabelEncoder()
timestamp_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
publishers = publisher_encoder.fit_transform(X_all['publisher'].astype(str)).reshape(-1,1)
timestamps = timestamp_encoder.fit_transform(X_all['timestamp']).reshape(-1,1)
categorical_features = np.hstack((publishers,timestamps))
one_hot_encoder.fit(categorical_features)
X_train_publisher = publisher_encoder.transform(train_data['publisher'].astype(str)).reshape(-1,1)
X_train_timestamp = timestamp_encoder.transform(train_data['timestamp']).reshape(-1,1)
X_train_categorical = one_hot_encoder.transform(np.hstack((X_train_publisher,X_train_timestamp)))
X_test_publisher = publisher_encoder.transform(test_data['publisher'].astype(str)).reshape(-1,1)
X_test_timestamp = timestamp_encoder.transform(test_data['timestamp']).reshape(-1,1)
X_test_categorical = one_hot_encoder.transform(np.hstack((X_test_publisher,X_test_timestamp)))




def get_text_features(data):
    text_features = list()
    for row in data.iterrows():
        words = list()
        title_text = row[1]['title']
        tokens = nltk.wordpunct_tokenize(title_text)
        tagged_tokens = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(t[0],pos=penn_to_wn(t[1])) for t in tagged_tokens]
        tokens = [w.lower() for w in tokens]
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w for w in tokens if not w in stop_words]

        words.extend(tokens)
        text_features.append(words)
        
    return text_features


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN


print ("Getting text features.....")
min_string_length = 50
lemmatizer = WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english') + list(string.punctuation)
vectorizer = HashingVectorizer(norm='l1') #this worked best
table = str.maketrans('', '', string.punctuation)
training_words = get_text_features(train_data)
test_words = get_text_features(test_data)
all_words = training_words + test_words

train_text = list()
test_text = list()
for item in training_words:
    text = " ".join(item)
    train_text.append(text)
for item in test_words:
    text = " ".join(item)
    test_text.append(text)
vectorizer.fit(train_text +test_text)
X_train_text_features = vectorizer.transform(train_text)
X_test_text_features= vectorizer.transform(test_text)

print ("Concatenating all features.....")


X_train_final = sp.hstack([X_train_categorical,X_train_text_features.astype(float)])
X_test_final = sp.hstack([X_test_categorical,X_test_text_features.astype(float)])


#print (X_train_final.shape)




#print (X_test_final.shape)




test_size = 0.15
random_state = 972
X_train, X_cv, y_train, y_cv = train_test_split(X_train_final, y_train_final, test_size = test_size, random_state = random_state)



dtrain = xgboost.DMatrix(X_train, label = y_train)
dtest = xgboost.DMatrix(X_cv,label = y_cv)



param = {
    'silent' : 1,
    'eta' : 0.05,
    'max_depth' : 5,
    'objective' : 'multi:softmax',
    'num_class' : 5
}
num_rounds = 1000
evallist = [(dtrain,'train'),(dtest,'cross_val')]
early_stopping_rounds=200
verbose_eval = False



#evaluations = xgboost.cv(param,dtrain,num_rounds, nfold=7,stratified=True,early_stopping_rounds=early_stopping_rounds,verbose_eval = verbose_eval)

print ("Training XGBoost Model for {:d} rounds with early stopping at {:d} rounds and learning rate {:.2f}.....".format(num_rounds,early_stopping_rounds,param['eta']))


model = xgboost.train(param,dtrain,num_rounds,evallist,early_stopping_rounds=early_stopping_rounds,verbose_eval = verbose_eval)


y_pred = model.predict(xgboost.DMatrix(X_cv),ntree_limit=model.best_iteration+1)
accuracy = accuracy_score(y_cv, y_pred)
print("Cross Validation Accuracy: %.2f%%" % (accuracy * 100.0))


print ("Outputing results file.....")


y_pred = model.predict(xgboost.DMatrix(X_test_final), ntree_limit=model.best_iteration+1)
submission = pd.DataFrame(test_data['article_id'])
submission['category'] = y_pred.astype(int)
submission.to_csv('results' + os.sep + 'xgboost_submission_{:.0f}.csv'.format(accuracy*10000),index=False)

print ("Done! Please find submission file in results/ folder")

