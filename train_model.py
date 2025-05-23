# train_model.py

import ast
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# 1) Check for GPU
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Training on CPU.")

# 2) Load your cleaned CSV
DATA_PATH = "processed_cleaned_data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cannot find {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())

# 3) Required columns
if not {"plot_summary", "genre_labels"}.issubset(df.columns):
    raise KeyError("Need 'plot_summary' and 'genre_labels' columns in CSV")

# 4) Parse the stringified label lists into real lists
def parse_labels(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

df["labels"] = df["genre_labels"].apply(parse_labels)
num_labels = len(df["labels"].iloc[0])
print(f"Parsed label vector length: {num_labels}")

# 5) Train/validation split
texts  = df["plot_summary"].astype(str).tolist()
labels = df["labels"].tolist()
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# 6) Tokenize
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
train_enc = tokenizer(
    X_train, padding="max_length", truncation=True, max_length=128
)
val_enc   = tokenizer(
    X_val,   padding="max_length", truncation=True, max_length=128
)

# 7) Dataset wrapper
class MovieDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = MovieDataset(train_enc, y_train)
val_dataset   = MovieDataset(val_enc,   y_val)

# 8) Model for multi-label classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

# 9) TrainingArguments — no evaluation kwargs
training_args = TrainingArguments(
    output_dir                 = "./results",
    num_train_epochs           = 3,
    per_device_train_batch_size= 8,
    per_device_eval_batch_size = 16,
    learning_rate              = 2e-5,
    weight_decay               = 0.01,
    save_strategy              ="epoch",
    save_total_limit           =2,
    logging_steps              = 100,    # prints loss every 100 steps
    save_steps                 = 1000,   # saves checkpoint every 1000 steps
    disable_tqdm               = False,  # show terminal tqdm bar
    report_to                  = None,   # disable TensorBoard
)

# 10) Trainer
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
)

# instead of trainer.train()
trainer.train()

# 12) Final evaluation (optional)
print("\nRunning final evaluation on validation set...")
metrics = trainer.evaluate()
print(metrics)

# 13) Save your fine-tuned artifacts
model.save_pretrained("genre_prediction_model_bert")
tokenizer.save_pretrained("genre_prediction_model_bert")
print("\n✅ Done! Model & tokenizer saved under ./genre_prediction_model_bert")
