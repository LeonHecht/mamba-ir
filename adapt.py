#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, Mamba2ForCausalLM, Mamba2Config
from torch.utils.data import Dataset
import pickle
import torch
# import transformers
# transformers.logging.set_verbosity_info()
# import logging
# logging.basicConfig(level=logging.INFO)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0


# In[2]:

import random

class LegalDataset(Dataset):
    def __init__(self, pickle_file_path, split="train", train_ratio=0.75, shuffle=True):
        """
        Initialize the dataset with pre-tokenized data from a pickle file.
        :param pickle_file_path: Path to the pickle file containing pre-tokenized data.
        :param split: "train" or "val" to specify the dataset split.
        :param train_ratio: Ratio of the dataset to use for training (default: 80%).
        """
        # Load pre-tokenized data from the pickle file
        with open(pickle_file_path, "rb") as f:
            pretokenized_data = pickle.load(f)

        # Shuffle the data if specified
        if shuffle:
            random.shuffle(pretokenized_data)
        
        # Split into train and validation
        train_size = int(len(pretokenized_data) * train_ratio)
        if split == "train":
            self.pretokenized_data = pretokenized_data[:train_size]
        elif split == "val":
            self.pretokenized_data = pretokenized_data[train_size:]
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.pretokenized_data)

    def __getitem__(self, idx):
        # Retrieve the pretokenized data
        data = self.pretokenized_data[idx]
        return {
            "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(data["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(data["input_ids"], dtype=torch.long),  # CLM uses input_ids as labels
        }


# In[3]:


# Load the pretrained tokenizer and model
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


# In[ ]:


# Path to the pickle file containing pre-chunked text
pickle_file_path = "tokenized_diverse_corpus_300M.pkl"

# Load the full train and validation datasets
train_dataset = LegalDataset(pickle_file_path, split="train")
val_dataset = LegalDataset(pickle_file_path, split="val")

# # Slice the datasets to use only half
# train_dataset_size = len(train_dataset) // 2
# val_dataset_size = len(val_dataset) // 2

# train_dataset = torch.utils.data.Subset(train_dataset, range(train_dataset_size))
# val_dataset = torch.utils.data.Subset(val_dataset, range(val_dataset_size))

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


# In[5]:


import math

def compute_metrics(eval_preds):
    """
    Compute perplexity from the evaluation loss.
    Hugging Face's Trainer provides eval_preds in the form of (predictions, labels),
    but for language models, it may only provide predictions or logits.

    Args:
        eval_preds: Tuple of (logits, labels) or (loss, None) depending on Trainer args.

    Returns:
        dict: Dictionary containing perplexity and other metrics (if added).
    """
    # Hugging Face often uses the eval_loss directly if available
    if isinstance(eval_preds, tuple):  # Check if it's a tuple
        logits, labels = eval_preds
        # Other metric computation can go here (if needed)
    else:
        # When the Trainer computes evaluation loss directly
        eval_loss = eval_preds
        perplexity = math.exp(eval_loss)
        return {"perplexity": perplexity}


# In[ ]:
results_dir = "./results_300M_diverse_shuffle_75train"

# Define training arguments
training_args = TrainingArguments(
    output_dir=results_dir,  # Directory for model checkpoints and logs
    overwrite_output_dir=False,
    num_train_epochs=3,  # Adjust based on your data size
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    gradient_accumulation_steps=8,  # Helps with small batch sizes
    eval_strategy="steps",  # Evaluate periodically
    eval_steps=320,  # Adjust as needed
    save_steps=320,  # Save model checkpoints periodically
    save_total_limit=2,  # Keep only the last two checkpoints
    logging_dir="./logs",  # Directory for training logs
    logging_steps=19,  # Log training progress every 50 steps
    learning_rate=1e-5,  # Lower LR for fine-tuning
    weight_decay=0.01,  # Regularization
    warmup_steps=1000,  # Gradual learning rate warm-up
    # fp16=True,  # Use mixed precision if supported
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)


# In[ ]:


# Start training
trainer.train()

# Save the final model
trainer.save_model(results_dir + "/mamba-130m-spanish-legal-300M-tokens-diverse")
tokenizer.save_pretrained(results_dir + "/mamba-130m-spanish-legal-300M-tokens-diverse")

