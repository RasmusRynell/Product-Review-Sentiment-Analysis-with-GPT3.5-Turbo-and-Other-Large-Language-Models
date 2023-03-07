import json
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score
import torch
import time
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
import transformers
import datasets
from datasets import load_metric
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from data_processing import *

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average=None)

    # Make all json serializable
    accuracy = float(accuracy)
    recall = float(recall)
    precision = float(precision)
    f1 = f1.tolist()
    avr_f1 = float(np.mean(f1))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "avr_f1": avr_f1}

if __name__ == '__main__':
    seed = 42
    transformers.enable_full_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    over_sample = False

    for over_sample in [False, True]:
        over_sample_string = "over_sample" if over_sample else "no_over_sample"

        cleaned_data = read_clean_data()
        train_data, validation_data, _ = split_data(cleaned_data, random_state=seed, validation=True, over_sample_train=over_sample)

        model_name = "distilbert-base-uncased"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        train_encodings = tokenizer(train_data['Summary'].tolist(), truncation=True, padding=True)
        validation_encodings = tokenizer(validation_data['Summary'].tolist(), truncation=True, padding=True)

        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = SentimentDataset(train_encodings, train_data['Sentiment'].tolist())
        validation_dataset = SentimentDataset(validation_encodings, validation_data['Sentiment'].tolist())

        if over_sample:
            training_args = TrainingArguments(
                output_dir=f'./models/checkpoints/{model_name}_{over_sample_string}/',
                eval_steps=50,
                learning_rate=2e-4,
                num_train_epochs=15,
                per_device_train_batch_size=128,
                per_device_eval_batch_size=128*4,
                fp16=True,
                warmup_steps=250,
                weight_decay=0.02,
                logging_steps=25,
                adam_beta1=0.89,
                adam_beta2=0.995,
                evaluation_strategy=IntervalStrategy.STEPS,
                metric_for_best_model = 'eval_loss',
                load_best_model_at_end=True,
                push_to_hub=False
            )

        else:
            training_args = TrainingArguments(
                output_dir=f'./models/checkpoints/{model_name}_{over_sample_string}/',
                eval_steps=50,
                learning_rate=1.5e-4,
                num_train_epochs=15,
                per_device_train_batch_size=128,
                per_device_eval_batch_size=128*4,
                fp16=True,
                warmup_steps=250,
                weight_decay=0.02,
                logging_steps=25,
                adam_beta1=0.91,
                adam_beta2=0.997,
                evaluation_strategy=IntervalStrategy.STEPS,
                metric_for_best_model = 'eval_loss',
                load_best_model_at_end=True,
                push_to_hub=False
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=25)]
        )

        trainer.evaluate()
        trainer.train()
        trainer.save_model(f"models/my/{model_name}_{over_sample_string}_done")

        # Save the log history to plot later
        with open(f"models/my/{model_name}_log_history_{over_sample_string}.json", "w") as f:
            json.dump(trainer.state.log_history, f)