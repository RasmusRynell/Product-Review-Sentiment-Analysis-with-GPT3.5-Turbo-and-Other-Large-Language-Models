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
import common

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

epochs = 10
batch_size = 8
learning_rate = 2e-5
epsilon = 1e-8


# This file contains tests for LLM models for different sentiment analysis datasets
# These models are english only, and a verity of small, medium and large models are tested
# They include:
# - bert-base-uncased, 110M parameters, english only, https://huggingface.co/bert-base-uncased
# - bert-large-uncased, 340M parameters, english only, https://huggingface.co/bert-large-uncased

# - RoBERTa-base, 125M parameters, english only, https://huggingface.co/roberta-base
# - RoBERTa-large, 355M parameters, english only, https://huggingface.co/roberta-large

# - DistilBert-base-uncased, 66M parameters, english only, https://huggingface.co/distilbert-base-uncased

# - albert-base-v2, 12M parameters, english only, https://huggingface.co/albert-base-v2
# - albert-large-v2, 18M parameters, english only, https://huggingface.co/albert-large-v2
# - albert-xlarge-v2, 60M parameters, english only, https://huggingface.co/albert-xlarge-v2
# - albert-xxlarge-v2, 235M parameters, english only, https://huggingface.co/albert-xxlarge-v2

# - xlnet-base-cased, 110M parameters, english only, https://huggingface.co/xlnet-base-cased
# - xlnet-large-cased, 340M parameters, english only, https://huggingface.co/xlnet-large-cased

# Fine-tuned models:
# - finbert, 110M parameters, english only, https://huggingface.co/ProsusAI/finbert
# - bertweet-base, 110M parameters, english only, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
# - bertweet-large, 340M parameters, english only, https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

non_fine_tuned_models = [
    # "bert-base-uncased", # Small, 110M parameters
    # "bert-large-uncased", # Large, 340M parameters
    # "bert-base-cased", # Small, 110M parameters
    # "bert-large-cased", # Large, 340M parameters
    # "roberta-base", # Small, 125M parameters
    # "roberta-large", # Large, 355M parameters
    # "distilbert-base-uncased", # Small, 66M parameters
    # "distilbert-base-cased", # Small, 66M parameters
    "albert-base-v2", # Small, 12M parameters
    # "albert-large-v2", # Medium, 18M parameters
    # "albert-xlarge-v2", # Large, 60M parameters, Works, Does not work (mem)
    # "albert-xxlarge-v2", # XLarge, 235M parameters, Does not work (mem)
    # "xlnet-base-cased", # Small, 110M parameters
    # "xlnet-large-cased", # Large, 340M parameters, Does not work (mem)
]

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_fn, epochs, device):
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


def fine_tune_model(model, tokenizer, train_df, validation_df):
    global epochs, batch_size, logging_steps, save_steps, output_dir, logging_dir, device, learning_rate, epsilon

    seed = common.seed

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure determinism and disable benchmarking for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tokenize and encode sequences
    train_encodings = tokenizer(train_df.x.tolist(), truncation=True, padding=True, return_tensors="pt")
    validation_encodings = tokenizer(validation_df.x.tolist(), truncation=True, padding=True, return_tensors="pt")

    # Create TensorDatasets
    contains_token_type_ids = "token_type_ids" in train_encodings
    if contains_token_type_ids:
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            train_encodings["token_type_ids"],
            torch.tensor(train_df.y.tolist())
        )
        validation_dataset = TensorDataset(
            validation_encodings["input_ids"],
            validation_encodings["attention_mask"],
            validation_encodings["token_type_ids"],
            torch.tensor(validation_df.y.tolist())
        )

    else:
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            torch.tensor(train_df.y.tolist())
        )
        validation_dataset = TensorDataset(
            validation_encodings["input_ids"],
            validation_encodings["attention_mask"],
            torch.tensor(validation_df.y.tolist())
        )

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * epochs

    # Create linear learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    train(model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_fn, epochs, device)



idx = 0

# For each dataset
for idx in range(0, 5):
    dataset_name, others = common.get_dataset(idx)
    df, train_df, validation_df, test_df = others

    # Reset all indexes
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print how long the train, validation and test sets are
    print(f'Train set length: {len(train_df)}', flush=True)
    print(f'Validation set length: {len(validation_df)}', flush=True)
    print(f'Test set length: {len(test_df)}', flush=True)

    # Print how many unique values there are in y_true
    print(f'Unique values in y_true: {len(train_df["y"].unique())}', flush=True)
    # Print how many of each unique value there are in y_true
    print(f'Unique values in y_true: {train_df["y"].value_counts().to_dict()}', flush=True)

    # Get number of labels
    num_labels = len(train_df["y"].unique())

    print(f'Number of labels: {num_labels}', flush=True)
    print(train_df["y"].unique(), flush=True)

    # Loop trough all different models, train and test them
    for model_name in non_fine_tuned_models:
        print(f"Model: {model_name}", flush=True)
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        #print(model.classifier.parameters, flush=True)
        model.to(device)

        # Get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=num_labels)

        # Fine-tune the model
        fine_tune_model(model, tokenizer, train_df, validation_df)
        
        # Test the model
        # test_model(model, tokenizer, test_df)