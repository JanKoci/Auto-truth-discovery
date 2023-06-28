#####################################
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
import pandas as pd
from bert_model import BertClassifier
from sklearn.metrics import classification_report


def get_merged_df(path):
    test_df = pd.read_csv(path, compression="gzip")
    label_map = {'real': 0, 'fake': 1}
    test_df['label'] = test_df['label'].map(label_map)
    test_df['label'] = test_df['label'].astype(int)

    # concatenate title and text columns
    test_df['text'] = test_df['title'] + ' ' + test_df['text']
    test_df['text'] = test_df['text'].astype(str)
    return test_df


def get_ood_df(path):
    ood = pd.read_csv(path, sep='\t')
    label_map = {'true': 0.0, 'fake': 1.0}
    ood['label'] = ood['label'].map(label_map)
    ood['label'] = ood['label'].astype(int)

    # concatenate title and text columns
    ood['text'] = ood['title'] + ' ' + ood['text']
    ood['text'] = ood['text'].astype(str)
    ood = ood.drop(columns=['title'])
    return ood

    
def test_report_fni_dataset(model:BertClassifier, path="../data/fni_dataset/fni.tsv"):
    test_df = pd.read_csv(path, sep='\t')
    label_map = {'true': 0.0, 'fake': 1.0}
    test_df['label'] = test_df['label'].map(label_map)
    test_df['label'] = test_df['label'].astype(int)

    # concatenate title and text columns
    test_df['text'] = test_df['title'] + ' ' + test_df['text']
    test_df['text'] = test_df['text'].astype(str)
    return model.test_report(test_df)

def test_report_merged(model:BertClassifier, path="../data/merged_dataset/merged.gzip"):
    merged = pd.read_csv(path, compression='gzip')
    label_map = {'real': 0.0, 'fake': 1.0}
    merged['label'] = merged['label'].map(label_map)
    merged['label'] = merged['label'].astype(int)

    # concatenate title and text columns
    merged['text'] = merged['title'] + ' ' + merged['text']
    merged['text'] = merged['text'].astype(str)
    
    predicted = model.predict(merged)
    report = classification_report(merged.label, predicted, target_names = ['0','1'])
    print(report)