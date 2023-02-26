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

from data import read_clean_data

def read_clean_data():
    data = pd.read_csv('sentiment.csv', encoding='latin-1')
    
    # Line (17301 in csv / 17299 in df) is corrupted, just remove it
    data = data.drop([17299])

    print(data.columns)

    # Drop columns that are not needed
    data = data.drop(['ProductPrice', 'Review', 'Rate', 'ProductName'], axis=1)

    data = data.dropna()

    # Convert to same values
    data['Sentiment'] = data['Sentiment'].replace(['Positive', 'Negative', 'Neutral'], ['positive', 'negative', 'neutral'])
    #data['Rate'] = data['Rate'].replace([1.0, 2.0, 3.0, 4.0, 5.0], ['1', '2', '3', '4', '5'])

    # Convert to ints
    data['Sentiment'] = data['Sentiment'].replace(['positive', 'neutral', 'negative'], [2, 1, 0])
    #data['Rate'] = data['Rate'].astype(int)

    # Drop text longer than 128 and shorter than 10
    data = data[data['Summary'].str.len() < 128]
    #data = data[data['Summary'].str.len() > 10]

    return data


def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy,}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cleaned_data = read_clean_data()

    # Split data into train, test and validation
    train_data = cleaned_data.sample(frac=0.8, random_state=42)
    test_data = cleaned_data.drop(train_data.index)
    validation_data = test_data.sample(frac=0.5, random_state=42)
    test_data = test_data.drop(validation_data.index)


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
        output_dir='./results',          # output directory
        learning_rate=2e-5,
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_strategy="epoch",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=validation_dataset,     # evaluation dataset
        #tokenizer=tokenizer,
        #data_collator=data_collator,
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
 

    trainer.train()
    trainer.evaluate()
    trainer.save_model("my_distilbert-base-uncased")

    # Print evaluation metrics
    print(trainer.evaluate())



    
