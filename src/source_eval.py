#####################################
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
import pandas as pd
import numpy as np
import evaluate
import torch
from bayes_model import MnbClassifier
from bert_model import BertClassifier
from  tldextract import extract


RELIABLE = 0
UNRELIABLE = 1


def get_registered_domain(df, source):
    df = df[df.source == source]
    url = df.iloc[0].url
    return extract(url).registered_domain

def get_src_label(df, source):
    src_label = df[df['source'] == source].label.values[0]
    return src_label

def get_common_sources(df, all_sources, sergio):
    registered_domains = [(src, get_registered_domain(df, src), get_src_label(df, src)) for src in all_sources]
    registered_domains = pd.DataFrame(registered_domains, columns=["source", "domain", "label"])
    # get only domains that appear in sergio's list
    registered_domains = registered_domains[registered_domains.domain.isin(sergio.keys())]
    return registered_domains.reset_index(drop=True)


def get_value_counts(predicted_labels):
    counts = np.unique(predicted_labels, return_counts=True)
    value_counts = dict(zip(counts[0], counts[1]))
    if 0 not in value_counts:
        value_counts[0] = 0
    elif 1 not in value_counts:
        value_counts[1] = 0
    return value_counts



def get_embedding(row, k, n):
    temp = row.embedding * n
    embedding_len = temp.shape[0]
    num_articles = row.num_articles
    if num_articles < embedding_len:
        if (num_articles < 2*k):
            embedding = temp.tolist()[:k] + temp.tolist()[k:2*k]
        else:
            diff = embedding_len - num_articles
            embedding = temp.tolist()[:k] + temp.tolist()[-(k+diff):-diff]
    else:
        embedding = temp.tolist()[:k] + temp.tolist()[-k:]
    return embedding


def get_train_test_embeddings(embeddings, common_sources, k, n=10000):
    train_embeddings = embeddings[~embeddings['source'].isin(common_sources)]
    test_embeddings = embeddings[embeddings['source'].isin(common_sources)]
    train_data = [(get_embedding(row, k, n), row.label) for i, row in train_embeddings.iterrows()]
    test_data = [(get_embedding(row, k, n), row.label) for i, row in test_embeddings.iterrows()]
    X_train = [x[0] for x in train_data]
    y_train = [x[1] for x in train_data]
    X_test = [x[0] for x in test_data]
    y_test = [x[1] for x in test_data]
    return X_train, y_train, X_test, y_test


def get_predicions(reliabilities, threshold=0.5):
    return [RELIABLE if r >= threshold else UNRELIABLE for r in reliabilities]

def evaluate_threshold(reliabilities, labels):
    result = []
    metric = evaluate.load("accuracy")
    for threshold in np.arange(0.1, 1, 0.1):
        predicted_labels = get_predicions(reliabilities, threshold)
        f1 = metric.compute(predictions=predicted_labels, references=labels)['accuracy']
        result.append((threshold, f1))
    return result



class BayesSourceEvaluator():
    # evaluate one source for MNB model 
    @staticmethod
    def eval_avg_prob(model:MnbClassifier, df:pd.DataFrame, sources:list):
        results = []
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            source_df = df[df['source'] == src]
            labels = source_df.label
            src_label = labels.values[0]
            probs = model.predict_proba(source_df)
            # probs_sum_normalized = np.sum(probs, axis=0) / num_articles
            probs_avg = np.mean(probs, axis=0)
            results.append((src, src_label, probs_avg))
        return results
    
    
    # evaluate one source for MNB model 
    @staticmethod
    def eval_agg_log_prob(model:MnbClassifier, df_test:pd.DataFrame):
        num_articles = df_test.shape[0]
        probs = model.predict_log_proba(df_test)
        probs_sum_normalized = np.sum(probs, axis=0) / num_articles
        return probs_sum_normalized
    
    # evaluate one source for MNB model using predicted labels
    @staticmethod
    def eval_by_predicted_labels(model:MnbClassifier, df_test:pd.DataFrame):
        num_articles = df_test.shape[0]
        predicted_labels = model.predict(df_test)
        value_counts = get_value_counts(predicted_labels)
        score = value_counts[UNRELIABLE] / num_articles
        return score, value_counts
    
    @staticmethod
    def eval_source_fake_n(model: MnbClassifier, df_test: pd.DataFrame, n:int=1):
        predicted_labels = model.predict(df_test)
        value_counts = get_value_counts(predicted_labels)
        if value_counts[UNRELIABLE] >= n:
            return 1
        else:
            return 0
        
    @staticmethod
    def eval_accuracy_for_all_sources(model: MnbClassifier, sources: list, df: pd.DataFrame):
        results = []
        metric = evaluate.load("accuracy")
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            source_df = df[df['source'] == src]
            labels = source_df.label
            src_label = labels.values[0]
            predicted = model.predict(source_df)
            accuracy = metric.compute(predictions=predicted, references=labels.values)['accuracy']
            results.append((src, accuracy, src_label, source_df.shape[0]))
        return results


class BertSourceEvaluator():
    # evaluate one source for bert
    # evaluation computed as sum of predictions for both labels divided by number of articles
    @staticmethod
    def evaluate_avg_prob(model:BertClassifier, df:pd.DataFrame, sources:list):
        results = []
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            source_df = df[df['source'] == src]
            num_articles = source_df.shape[0]
            res = model.predict(source_df)
            probabilities = torch.softmax(torch.tensor(res.predictions), dim=-1).numpy()
            preds_sum_normalized = np.sum(probabilities, axis=0) / num_articles
            results.append((src, preds_sum_normalized[0]))
        return results
    
        
    # evaluate one source for MNB model using predicted labels
    @staticmethod
    def eval_by_predicted_labels(model:BertClassifier, df:pd.DataFrame, sources:list):
        results = []
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            source_df = df[df['source'] == src]
            num_articles = source_df.shape[0]
            res = model.predict(source_df)
            predicted_labels = np.argmax(res.predictions, axis=-1)
            value_counts = get_value_counts(predicted_labels)
            score = value_counts[RELIABLE] / num_articles
            results.append((src, score))
        return results
    

    @staticmethod
    def get_prediction_value_counts(model:BertClassifier, df:pd.DataFrame, sources:list):
        results = []
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            temp_dict = {}
            source_df = df[df['source'] == src]
            src_label = source_df['label'].values[0]
            res = model.predict(source_df)
            predicted_labels = np.argmax(res.predictions, axis=-1)
            value_counts = get_value_counts(predicted_labels)
            temp_dict['source'] = src
            temp_dict['label'] = src_label
            temp_dict['value_counts'] = value_counts
            results.append(temp_dict)
        return pd.DataFrame(results, columns=['source', 'label', 'value_counts'])
    

    @staticmethod
    def eval_accuracy_for_all_sources(model:BertClassifier, sources: list, df: pd.DataFrame):
        results = []
        metric = evaluate.load("accuracy")
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            source_df = df[df['source'] == src]
            labels = source_df['label']
            src_label = labels.values[0]
            res = model.predict(source_df)
            predicted_labels = np.argmax(res.predictions, axis=-1)
            accuracy = metric.compute(predictions=predicted_labels, references=res.label_ids)['accuracy']
            results.append((src, accuracy, src_label, source_df.shape[0]))
        return results
    

    @staticmethod
    def get_k_best_and_worst(model:BertClassifier, sources:list, df:pd.DataFrame, k:int=5):
        results = []
        for i, src in enumerate(sources):
            print("Evaluating source: {0} {1}/{2}".format(src, i+1, len(sources)))
            temp_dict = {}
            source_df = df[df['source'] == src]
            src_label = source_df['label'].values[0]
            res = model.predict(source_df)

            probabilities = torch.softmax(torch.tensor(res.predictions), dim=-1)
            reliable_probabilities = probabilities[:, 0]
            len_reliable = len(reliable_probabilities)

            # get top and bottom k
            if (len_reliable <= k) or (len_reliable < 2*k):
                top_k = torch.topk(reliable_probabilities, k=len_reliable).values
                bottom_k = torch.zeros(2*k - len_reliable)
            else:
                top_k = torch.topk(reliable_probabilities, k=k).values
                bottom_k = torch.topk(reliable_probabilities, k=k, largest=False).values.flip(0)

            # concat top and bottom k
            embeddings = torch.cat((top_k, bottom_k), dim=0)

            # add new row to results
            temp_dict['source'] = src
            temp_dict['label'] = src_label
            temp_dict['embedding'] = embeddings
            temp_dict['num_articles'] = len_reliable
            results.append(temp_dict)

        return pd.DataFrame(results, columns=['source', 'label', 'embedding', 'num_articles'])
    