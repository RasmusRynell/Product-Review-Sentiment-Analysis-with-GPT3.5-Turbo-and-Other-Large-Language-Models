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
import datasets
from datasets import load_metric

from read_data import read_clean_data

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cleaned_data = read_clean_data()

    seed = 42
    train_data = cleaned_data.sample(frac=0.8, random_state=seed)
    test_data = cleaned_data.drop(train_data.index)
    validation_data = test_data

    # Fine tune "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    train_encodings = tokenizer(train_data['Summary'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_data['Summary'].tolist(), truncation=True, padding=True)
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
    test_dataset = SentimentDataset(test_encodings, test_data['Sentiment'].tolist())
    validation_dataset = SentimentDataset(validation_encodings, validation_data['Sentiment'].tolist())

    training_args = TrainingArguments(
        output_dir='./models/checkpoints/',
        learning_rate=2e-4,
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=250,                # number of warmup steps for learning rate scheduler
        weight_decay=0.02,               # strength of weight decay
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics
    )
 

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model("models/my/distilbert-base-uncased_done")

    print(trainer.state.log_history)
    # # Plot training loss
    # plt.plot(trainer.state.log_history["loss"])
    # plt.title("Training loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.savefig(f"plots/my_distilbert-base-uncased_done_loss.png")

    # # Plot training accuracy
    # plt.plot(trainer.state.log_history["eval_accuracy"])
    # plt.title("Training accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.savefig(f"plots/my_distilbert-base-uncased_done_accuracy.png")

    # # Plot scheduler learning rate
    # plt.plot(trainer.state.log_history["lr"])
    # plt.title("Scheduler learning rate")
    # plt.xlabel("Epoch")
    # plt.ylabel("Learning rate")
    # plt.savefig(f"plots/my_distilbert-base-uncased_done_lr.png")

    # # Plot training time
    # plt.plot(trainer.state.log_history["total_flos"])
    # plt.title("Training time")
    # plt.xlabel("Epoch")
    # plt.ylabel("Training time")
    # plt.savefig(f"plots/my_distilbert-base-uncased_done_time.png")



    
