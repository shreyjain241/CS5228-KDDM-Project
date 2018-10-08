# encoding=utf-8
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

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

def load_and_process(train_path, test_path, feat_transform='svd'):
    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_labels = train_data['category'].tolist()

    col_names = list(train_data)
    col_names.remove('article_id')
    col_names.remove('category')
    train_data['concat_feat'] = ""
    test_data['concat_feat'] = ""
    for col in col_names:
        train_data['concat_feat'] = train_data['concat_feat'].map(str) + train_data[col].map(str)
        test_data['concat_feat'] = test_data['concat_feat'].map(str) + test_data[col].map(str)

    print("Extracting features...")
    #train_data['concat_feat'] = train_data['concat_feat'].apply(lambda x : pre_process(x, porter))
    #test_data['concat_feat'] = test_data['concat_feat'].apply(lambda x : pre_process(x, porter))

    vectorizer = TfidfVectorizer().fit(train_data['concat_feat'])
    train_feats = vectorizer.transform(train_data['concat_feat'])
    train_feats = preprocessing.scale(train_feats, with_mean=False)
    test_feats = vectorizer.transform(test_data['concat_feat'])
    test_feats = preprocessing.scale(test_feats, with_mean=False)

    if(feat_transform == 'svd'):
        svd = TruncatedSVD(n_components=2000, n_iter=2).fit(train_feats)
        train_feats = svd.transform(train_feats)
        test_feats = svd.transform(test_feats)

    elif(feat_transform == 'lda'):
        lda = LatentDirichletAllocation(n_components=2000, learning_method='batch').fit(train_feats)
        train_feats = lda.transform(train_feats)
        test_feats = lda.transform(test_feats)

    return (train_feats, train_labels, test_feats)
