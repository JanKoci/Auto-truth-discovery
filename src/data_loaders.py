#####################################
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
import nela_helpers as nh
import pandas as pd

class DistilBertLoader():
    
    @staticmethod
    def load_test_data(path,
                       compression="gzip",
                       apply_preprocessing=True,
                       filter_mixed=True):
        df = nh.read_nela_df(path, compression)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        if filter_mixed:
            df = df[df['label'] != 2]
        if apply_preprocessing:
            df['text'] = df['text'].apply(nh.TextPreprocess.remove_html)
            df['text'] = df['text'].apply(nh.TextPreprocess.remove_symbols)
        return df
    
    @staticmethod
    def load_train_data(path,
                        compression="gzip",
                        filter_mixed=True):
        df = nh.read_nela_df(path, compression)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        if filter_mixed:
            df = df[df['label'] != 2]
        return df
    
    @staticmethod
    def load_merged_dataset(path, compression="gzip"):
        merged = pd.read_csv(path, compression=compression)
        label_map = {'real': 0.0, 'fake': 1.0}
        merged['label'] = merged['label'].map(label_map)
        merged['label'] = merged['label'].astype(int)

        # concatenate title and text columns
        merged['text'] = merged['title'] + ' ' + merged['text']
        merged['text'] = merged['text'].astype(str)
        merged['text'] = merged['text'].apply(nh.TextPreprocess.remove_html)
        return merged


class BayesLoader():

    @staticmethod
    def load_test_data(path, 
                       compression="gzip", 
                       apply_preprocessing=True,
                       filter_mixed=True):
        df = nh.read_nela_df(path, compression)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        if filter_mixed:
            df = df[df['label'] != 2]
        if apply_preprocessing:
            df['text'] = df['text'].apply(nh.TextPreprocess.remove_html)
            df['text'] = df['text'].apply(nh.TextPreprocess.remove_stopwords)
        return df
    
    @staticmethod
    def load_train_data(path,
                        compression="gzip",
                        filter_mixed=True):
        df = nh.read_nela_df(path, compression)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        if filter_mixed:
            df = df[df['label'] != 2]
        df['text'] = df['text'].apply(nh.TextPreprocess.remove_stopwords)
        return df
    
    @staticmethod
    def load_merged_dataset(path, compression="gzip"):
        merged = pd.read_csv(path, compression=compression)
        label_map = {'real': 0.0, 'fake': 1.0}
        merged['label'] = merged['label'].map(label_map)
        merged['label'] = merged['label'].astype('float64')

        # concatenate title and text columns
        merged['text'] = merged['title'] + ' ' + merged['text']
        merged['text'] = merged['text'].astype(str)
        merged['text'] = merged['text'].apply(nh.TextPreprocess.remove_html)
        merged['text'] = merged['text'].apply(nh.TextPreprocess.remove_stopwords)
        
        return merged
    
    def load_fni_dataset(path, sep="\t"):
        df = pd.read_csv(path, sep=sep)
        df.fillna('Unknown', inplace=True)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype('float64')
        df['text'] = df['text'].apply(nh.TextPreprocess.remove_stopwords)
        return df