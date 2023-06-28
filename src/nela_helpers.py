#####################################
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
import re,string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bayes_model import MnbClassifier


class TextPreprocess():
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    @staticmethod
    def remove_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the stopwords from text
    @staticmethod
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in TextPreprocess.stop:
                final_text.append(i.strip())
        return " ".join(final_text)


    # remove @ symbols from text
    @staticmethod
    def remove_symbols(text):
        text = re.sub('@', '', text)
        # remove multiple spaces
        return ' '.join(text.split())


    @staticmethod
    def create_and_filter_text(df):    
        # concatenate title and text columns
        df['text'] = df['title'] + ' ' + df['content']
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].apply(TextPreprocess.remove_html)
        df['text'] = df['text'].apply(TextPreprocess.remove_stopwords)
        df['text'] = df['text'].apply(TextPreprocess.remove_symbols)
        return df


def read_nela_df(path, compresion='gzip'):
    df = pd.read_csv(path, compression=compresion)
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    df = df[df['text'].notna()]
    df = df[df['label'].notna()]
    return df


def get_sources_with_less_than_n_articles(df, n):
    sources_counts = df.source.value_counts()
    src_new = sources_counts[sources_counts < n].index.tolist()
    df = df[df['source'].isin(src_new)]


def random_drop(df, n=100000, label=0.0):
    np.random.seed(42)
    drop_indices = np.random.choice(df[df.label == label].index, n, replace=False)
    return df.drop(drop_indices)


def interpret_mnb(mnb, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()
    true = mnb.feature_log_prob_[0]
    fake = mnb.feature_log_prob_[1]

    true = [(value, feature_names[i]) for i, value in enumerate(true)]
    fake = [(value, feature_names[i]) for i, value in enumerate(fake)]

    true = sorted(true, reverse=True)
    fake = sorted(fake, reverse=True)

    return true[:n], fake[:n]


# interpret the importance of words (x) as P(x|True) / P(x|Fake)
# the smaller the value the more likely it is to be true and vice versa
def interpret_mnb_ratio(mnb, vectorizer, n=30):
    feature_names = vectorizer.get_feature_names_out()
    true = mnb.feature_log_prob_[0]
    fake = mnb.feature_log_prob_[1]

    ratio = [(true_prob / fake_prob) for (true_prob, fake_prob) in zip(true, fake)]

    zipped = list(zip(ratio, feature_names))
    
    true_ratio = sorted(zipped)
    fake_ratio = sorted(zipped, reverse=True)

    return true_ratio[:n], fake_ratio[:n]


def test_report_fni_dataset(model:MnbClassifier, path="../data/fni_dataset/fni.tsv"):
    test_df = pd.read_csv(path, sep='\t')
    label_map = {'true': 0.0, 'fake': 1.0}
    test_df['label'] = test_df['label'].map(label_map)
    test_df['label'] = test_df['label'].astype('float64')

    # concatenate title and text columns
    test_df['text'] = test_df['title'] + ' ' + test_df['text']
    test_df['text'] = test_df['text'].astype(str)

    test_df['text'] = test_df['text'].apply(TextPreprocess.remove_html)
    test_df['text'] = test_df['text'].apply(TextPreprocess.remove_stopwords)
    
    predicted = model.predict(test_df)
    report = classification_report(test_df.label, predicted, target_names = ['0','1'])
    print(report)

def test_report_merged(model:MnbClassifier, path="../data/merged_dataset/merged.gzip"):
    merged = pd.read_csv(path, compression='gzip')
    label_map = {'real': 0.0, 'fake': 1.0}
    merged['label'] = merged['label'].map(label_map)
    merged['label'] = merged['label'].astype('float64')

    # concatenate title and text columns
    merged['text'] = merged['title'] + ' ' + merged['text']
    merged['text'] = merged['text'].astype(str)

    merged['text'] = merged['text'].apply(TextPreprocess.remove_html)
    merged['text'] = merged['text'].apply(TextPreprocess.remove_stopwords)
    
    predicted = model.predict(merged)
    report = classification_report(merged.label, predicted, target_names = ['0','1'])
    print(report)


def filter_out_sources(df, sources):
    df = df[~df['source'].isin(sources)]
    return df


def test_report(labels, predicted, target_names=['0','1']):
    report = classification_report(labels, predicted, target_names = target_names, zero_division=0)
    print(report)
    return report


def read_keywords(filepath='nela_keywords.txt'):
    with open(filepath, 'r') as f:
        keywords = f.read().splitlines()
    return keywords

def read_keywords_set(filepath='nela_keywords.txt'):
    return set(read_keywords(filepath))


def remove_keywords(text, keywords):
    resultwords  = [word for word in re.split("\W+", text) if word and word.lower() not in keywords]
    return ' '.join(resultwords)


def filter_keywords(df, keywords):
    df['text'] = df['text'].apply(lambda text: remove_keywords(text, keywords))
    return df