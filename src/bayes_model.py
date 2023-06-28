#####################################
# This file contains the Multinomial Naive Bayes classifier
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
import nela_helpers as nh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class MnbClassifier():
    def __init__(self, ngram_range=(1,1)) -> None:
        # Create a Multinomial Naive Bayes classifier
        self.classifier = MultinomialNB()
        self.tfidf = TfidfVectorizer(use_idf=True, ngram_range=ngram_range)

    def fit(self, train_df):
        # Train the classifier
        text = train_df['text']
        X = self.tfidf.fit_transform(text)
        y = train_df['label']
        self.classifier.fit(X, y)
        print("Training complete")

    def predict(self, test_df: pd.DataFrame):
        # Predict the labels
        text = test_df['text']
        X = self.tfidf.transform(text)
        return self.classifier.predict(X)
    
    def predict_text(self, text: str):
        # Predict the labels
        X = self.tfidf.transform([text])
        return self.classifier.predict(X)
    
    def predict_proba(self, test_df):
        # Predict the probabilities
        text = test_df['text']
        X = self.tfidf.transform(text)
        return self.classifier.predict_proba(X)
    
    def predict_proba_text(self, text: str):
        # Predict the probabilities
        X = self.tfidf.transform([text])
        return self.classifier.predict_proba(X)
    
    def predict_log_proba(self, test_df):
        # Predict the log probabilities
        text = test_df['text']
        X = self.tfidf.transform(text)
        return self.classifier.predict_log_proba(X)
    
    def test_report(self, test_df):
        y_pred = self.predict(test_df)
        y_true = test_df['label'].values
        nh.test_report(y_true, y_pred)
