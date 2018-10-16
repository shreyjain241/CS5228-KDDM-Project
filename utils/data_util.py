# encoding=utf-8
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>', re.S)
    return html_prog.sub('', str)

def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)

def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)

def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)

def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)

def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)

def replace_emoticon(emoticon_dict, str):
    for k, v in emoticon_dict.items():
        str = str.replace(k, v)
    return str

def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]', '', current_tweet)

def pre_process(str, porter):
    # do not change the preprocessing order only if you know what you're doing
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)
    str = rm_punctuation(str)

    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            #print(str)
            pass
    except:
        #print(str)
        pass

    return str

def load_and_process(train_path, test_path, feat_transform='svd', split=0.8, vectorizer='hashing'):
    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_labels = train_data['category']

    col_names = list(train_data)
    col_names.remove('article_id')
    col_names.remove('category')

    train_data['concat_feat'] = ""
    test_data['concat_feat'] = ""
    for col in col_names:
        train_data['concat_feat'] = train_data['concat_feat'].map(str) + train_data[col].map(str)
        test_data['concat_feat'] = test_data['concat_feat'].map(str) + test_data[col].map(str)

    print("Preprocessing text...")
    train_data['concat_feat'] = train_data['concat_feat'].apply(lambda x : pre_process(x, porter))
    test_data['concat_feat'] = test_data['concat_feat'].apply(lambda x : pre_process(x, porter))

    print("Creating random train/val split...")
    total_count = len(train_data['concat_feat'])
    mask = np.random.rand(total_count) < split
    X_train = train_data['concat_feat'][mask]
    X_val = train_data['concat_feat'][~mask]
    y_train = train_labels[mask].tolist()
    y_val = train_labels[~mask].tolist()

    if(vectorizer == 'hashing'):
        print("Performing the hashing...")
        hv = HashingVectorizer(n_features=1500)
        X_train = hv.transform(X_train)
        X_val = hv.transform(X_val)
        X_test = hv.transform(test_data['concat_feat'])

    elif(vectorizer == 'tfidf'):
        print("Getting tf-idf weights...")
        vectorizer = TfidfVectorizer().fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_train = preprocessing.scale(X_train, with_mean=False)
        X_val = vectorizer.transform(X_val)
        X_val = preprocessing.scale(X_val, with_mean=False)
        X_test = vectorizer.transform(test_data['concat_feat'])
        X_test = preprocessing.scale(X_test, with_mean=False)

    if(feat_transform == 'svd'):
        print("Perfoming svd based reduction...")
        svd = TruncatedSVD(n_components=2000, n_iter=2).fit(X_train)
        X_train = svd.transform(X_train)
        X_val = svd.transform(X_val)
        X_test = svd.transform(X_test)

    elif(feat_transform == 'lda'):
        print("Perfoming LDA based reduction...")
        lda = LatentDirichletAllocation(n_components=2000, learning_method='batch').fit(X_train)
        X_train = lda.transform(X_train)
        X_val = lda.transform(X_val)
        X_test = lda.transform(X_test)

    elif(feat_transform == 'chi2'):
        print("Perfoming chi2 feature selection...")
        chi2_obj = SelectKBest(chi2, k=2000).fit(X_train, y_train)
        X_train = chi2_obj.transform(X_train)
        X_val = chi2_obj.transform(X_val)
        X_test = chi2_obj.transform(X_test)

    elif(feat_transform == 'linearDA'):
        print("Perfoming linear discriminant feat selection...")
        linearDA = LinearDiscriminantAnalysis(n_components=1000).fit(X_train.toarray(), y_train)
        X_train = linearDA.transform(X_train)
        X_val = linearDA.transform(X_val)
        X_test = linearDA.transform(X_test)

    return (X_train, y_train, X_val, y_val, X_test)