from transformers import pipeline
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
import numpy as np
import random
import matplotlib.pyplot as plt
import kaggle


data = [('movie','akshaypawar7/millions-of-movies', 'movies.csv', 'overview', 'budget'),
        ('musk','marta99/elon-musks-tweets-dataset-2022', 'cleandata.csv', 'Cleaned_Tweets', 'Likes')]

curr_data = data[0] # Can be changed

try:
    kaggle.api.authenticate()
    for dataset in data:
        print(f"Downloading {dataset[0]} dataset")
        kaggle.api.dataset_download_files(dataset[1], path=f'data/{dataset[0]}', unzip=True)
except:
    print("Kaggle API not working, either download the data manually or use the Kaggle CLI")
    print(f"Download the data from {address} and place it in the data/{folder} folder")


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def clean_up_dataset(df, df_col_x, df_col_y, drop_zeros=False):
    # Drop rows with missing values in the df_col_x or df_col_y columns
    df = df.dropna(subset=[df_col_x, df_col_y])
    # Drop rows with zero values in the df_col_y column
    if drop_zeros:
        df = df[df[df_col_y] != 0]

    # Convert the df_col_y column to float
    df[df_col_y] = df[df_col_y].astype(float)

    # Convert to lowercase
    df[df_col_x] = df[df_col_x].str.lower()

    return df[[df_col_x, df_col_y]]


def split_data(df, test_size, validation_size, seed):
    train_size = 1 - test_size - validation_size

    # Split the data into train test and validation sets
    train_df, test_val_df = train_test_split(df, train_size=train_size, random_state=seed)
    validation_df, test_df = train_test_split(test_val_df, train_size=validation_size/(test_size+validation_size), random_state=seed)
    
    return train_df, validation_df, test_df

# Read data
text_column = curr_data[3]
ans_column = curr_data[4]
df = pd.read_csv(f'data/{curr_data[0]}/{curr_data[2]}')
df = clean_up_dataset(df, text_column, ans_column, drop_zeros=True)


# Reduce dataset by 80% for faster training
df = df.sample(frac=0.01, random_state=seed)

train_df, validation_df, test_df = split_data(df, test_size=0.2, validation_size=0.2, seed=seed)
print(f"Train set size: \t{len(train_df)} ({len(train_df)/len(df)*100:.2f}%", flush=True)
print(f"Validation set size: \t{len(validation_df)} ({len(validation_df)/len(df)*100:.2f}%", flush=True)
print(f"Test set size: \t\t{len(test_df)} ({len(test_df)/len(df)*100:.2f}%", flush=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Tokenize the data
encoded_corpus_train = tokenizer(train_df[text_column].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids_train = encoded_corpus_train['input_ids']
attention_mask_train = encoded_corpus_train['attention_mask']

encoded_corpus_validation = tokenizer(validation_df[text_column].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids_validation = encoded_corpus_validation['input_ids']
attention_mask_validation = encoded_corpus_validation['attention_mask']

encoded_corpus_test = tokenizer(test_df[text_column].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids_test = encoded_corpus_test['input_ids']
attention_mask_test = encoded_corpus_test['attention_mask']

# Scale the data
y_scaler = StandardScaler()
y_scaler.fit(train_df[ans_column].values.reshape(-1, 1))
train_labels = y_scaler.transform(train_df[ans_column].values.reshape(-1, 1))
validation_labels = y_scaler.transform(validation_df[ans_column].values.reshape(-1, 1))
test_labels = y_scaler.transform(test_df[ans_column].values.reshape(-1, 1))

# Create the dataset with float32
train_dataset = TensorDataset(input_ids_train, attention_mask_train, torch.tensor(train_labels, dtype=torch.float32))
validation_dataset = TensorDataset(input_ids_validation, attention_mask_validation, torch.tensor(validation_labels, dtype=torch.float32))
test_dataset = TensorDataset(input_ids_test, attention_mask_test, torch.tensor(test_labels, dtype=torch.float32))

# Create the data loaders
batch_size = 3
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



# Create the model
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=1)
model.to(device)

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

# Create the learning rate scheduler
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Create the loss function
loss_fn = nn.MSELoss()

# Train the model
def train(model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_fn, epochs, device):
    # Store the loss and accuracy for plotting
    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        print("-"*10, flush=True)

        # Training
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            # Move batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Clear any previously calculated gradients
            model.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            # Backward pass
            loss.backward()

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

            # Update training loss
            train_loss += loss.item()

        # Calculate the average loss over the training data
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        print(f"Training loss: {avg_train_loss}", flush=True)

        # Validation
        model.eval()
        validation_loss = 0
        for batch in validation_dataloader:
            # Move batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Forward pass
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]

            # Update validation loss
            validation_loss += loss.item()

        # Calculate the average loss over the validation data
        avg_validation_loss = validation_loss / len(validation_dataloader)
        validation_losses.append(avg_validation_loss)

        print(f"Validation loss: {avg_validation_loss}", flush=True)
        
    return train_losses, validation_losses

train_losses, validation_losses = train(model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_fn, epochs, device)

# Plot the loss
plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend()
plt.show()

# Evaluate the model
def evaluate(model, test_dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    for batch in test_dataloader:
        # Move batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

        # Update test loss
        test_loss += loss.item()

    # Calculate the average loss over the test data
    avg_test_loss = test_loss / len(test_dataloader)

    print(f"Test loss: {avg_test_loss}", flush=True)

evaluate(model, test_dataloader, loss_fn, device)
