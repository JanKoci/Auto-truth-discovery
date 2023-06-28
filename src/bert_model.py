#####################################
# This file contains the BERT and DistilBERT classifiers
# Master's thesis: Automated truth discovery
# Author: Jan Koci
# Date: 05-05-2023
####################################
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
import nela_helpers as nh
import torch


class BertClassifier():
    def __init__(self, model_name="bert-base-uncased") -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = device
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        self.__trainer = Trainer(model=self.__model, tokenizer=self.__tokenizer)

    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def model(self):
        return self.__model    

    def __tokenize_function(self, examples):
        return self.__tokenizer(examples["text"], truncation=True, padding="max_length")

    def train(self, df_train:pd.DataFrame, df_test:pd.DataFrame, output_dir="./results"):
        data_collator = DataCollatorWithPadding(tokenizer=self.__tokenizer)
        dataset_train = Dataset.from_pandas(df_train)
        dataset_test = Dataset.from_pandas(df_test)

        train_tokenized = dataset_train.map(self.__tokenize_function, batched=True)
        test_tokenized = dataset_test.map(self.__tokenize_function, batched=True)

        train_tokenized.set_format("torch")
        test_tokenized.set_format("torch")

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.__model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            tokenizer=self.__tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        self.__trainer = trainer

        best_run = trainer.train()
        return best_run
    
    def predict(self, df:pd.DataFrame):
        dataset = Dataset.from_pandas(df)
        tokenized = dataset.map(self.__tokenize_function, batched=True)
        tokenized.set_format("torch")
        predictions = self.__trainer.predict(tokenized)
        return predictions
    

    def test_report(self, df:pd.DataFrame):
        preds = self.predict(df)
        predictions = np.argmax(preds.predictions, axis=-1)
        true = preds.label_ids
        return nh.test_report(true, predictions, target_names=["0", "1"])
    


class DistilBertClassifier():
    def __init__(self, model_name="distilbert-base-cased") -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = device
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        self.__trainer = Trainer(model=self.__model, tokenizer=self.__tokenizer)

    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def model(self):
        return self.__model
    

    def __tokenize_function(self, examples):
        return self.__tokenizer(examples["text"], truncation=True, padding="max_length")

    def train(self, df_train:pd.DataFrame, df_test:pd.DataFrame, output_dir="./results"):
        data_collator = DataCollatorWithPadding(tokenizer=self.__tokenizer)
        dataset_train = Dataset.from_pandas(df_train)
        dataset_test = Dataset.from_pandas(df_test)

        train_tokenized = dataset_train.map(self.__tokenize_function, batched=True)
        test_tokenized = dataset_test.map(self.__tokenize_function, batched=True)

        train_tokenized.set_format("torch")
        test_tokenized.set_format("torch")

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.__model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            tokenizer=self.__tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        self.__trainer = trainer

        best_run = trainer.train()
        return best_run
    
    def predict(self, df:pd.DataFrame):
        dataset = Dataset.from_pandas(df)
        tokenized = dataset.map(self.__tokenize_function, batched=True)
        tokenized.set_format("torch")
        predictions = self.__trainer.predict(tokenized)
        return predictions
    

    def test_report(self, df:pd.DataFrame):
        preds = self.predict(df)
        predictions = np.argmax(preds.predictions, axis=-1)
        true = preds.label_ids
        return nh.test_report(true, predictions, target_names=["0", "1"])