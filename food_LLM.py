from transformers import pipeline
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import kaggle
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 10
batch_size = 10
learning_rate = 2e-5
epsilon = 1e-8

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def run(train_df, validation_df, test_df):
    global device, epochs, batch_size, learning_rate, epsilon

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # Tokenize the text
    train_encodings = tokenizer(train_df['review'].tolist(), truncation=True, padding=True)
    validation_encodings = tokenizer(validation_df['review'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_df['review'].tolist(), truncation=True, padding=True)

    # Create the dataset
    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(train_df['rating'].tolist()))
    validation_dataset = TensorDataset(torch.tensor(validation_encodings['input_ids']), torch.tensor(validation_encodings['attention_mask']), torch.tensor(validation_df['rating'].tolist()))
    test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(test_df['rating'].tolist()))

    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    # Create the scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Create the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Move the model to the GPU
    model = model.to(device)

    # Train the model
    train_losses = []
    validation_losses = []
    validation_accuracy = []

    print("Training model", flush=True)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        i = 0
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2 if len(batch) == 3 else 3] # If the model has a token_type_ids argument, the third element of batch is token_type_ids
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch + 1}/{epochs} Batch {i}/{len(train_dataloader)}", flush=True, end="\r")
            i += 1

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        predictions, true_vals = [], []
        for batch in validation_dataloader:
            batch = tuple(b.to(device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2 if len(batch) == 3 else 3] # If the model has a token_type_ids argument, the third element of batch is token_type_ids
            }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            val_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        val_loss /= len(validation_dataloader)
        validation_losses.append(val_loss)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        accuracy = flat_accuracy(predictions, true_vals)
        validation_accuracy.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        print(f"Training loss: {train_loss}", flush=True)
        print(f"Validation loss: {val_loss}", flush=True)
        print(f"Validation accuracy: {accuracy}", flush=True)

        # Confusion matrix
        print("Confusion matrix:")
        print(confusion_matrix(true_vals, np.argmax(predictions, axis=1)), flush=True)


    return train_losses, validation_losses