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

# def compute_metrics(eval_pred):
#    load_accuracy = load_metric("accuracy")
  
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
#    return {"accuracy": accuracy}

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

    # Print
    print(f"Accuracy: {accuracy}", flush=True)
    print(f"Recall: {recall}", flush=True)
    print(f"Precision: {precision}", flush=True)
    print(f"F1: {f1}", flush=True)
    print(f"Average F1: {avr_f1}", flush=True)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "avr_f1": avr_f1}

if __name__ == '__main__':
    seed = 42
    transformers.enable_full_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cleaned_data = read_clean_data()
    train_data, validation_data, _ = split_data(cleaned_data, random_state=seed, validation=True, over_sample_train=True)

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

    training_args = TrainingArguments(
        output_dir=f'./models/checkpoints/{model_name}/',
        save_total_limit = 5,
        eval_steps=500,
        learning_rate=2e-4,
        num_train_epochs=4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=250,                # number of warmup steps for learning rate scheduler
        weight_decay=0.02,               # strength of weight decay
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.STEPS,
        metric_for_best_model = 'avr_f1',
        load_best_model_at_end=True,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )


    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(f"models/my/{model_name}_done")

    #print(trainer.state.log_history)
    # Save the log history
    with open(f"models/my/{model_name}_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)