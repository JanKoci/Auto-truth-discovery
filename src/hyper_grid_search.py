#####################################
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
import sys


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "num_epochs": trial.suggest_int("num_epochs", 2, 4),
    }

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to("cuda")


train_file = sys.argv[1]
test_file = sys.argv[2]
df_train = nh.read_nela_df(train_file)
df_test = nh.read_nela_df(test_file)
df_train['label'] = df_train['label'].astype(int)
df_test['label'] = df_test['label'].astype(int)


# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to("cuda")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

train_tokenized = dataset_train.map(tokenize_function, batched=True)
test_tokenized = dataset_test.map(tokenize_function, batched=True)

train_tokenized.set_format("torch")
test_tokenized.set_format("torch")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    print("compute_metrics called")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    temp = pd.DataFrame({'predictions': predictions, 'labels': labels})
    temp.to_csv('compute_metrics.csv')
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results_hyperopt",
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20
)

print(best_trial)
print(best_trial.hyperparameters)
