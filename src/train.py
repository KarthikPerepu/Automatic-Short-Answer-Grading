import os
# Stop huggingface tokenizers from deadlocking HPC/SLURM environments
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, cohen_kappa_score

def compute_qwk(y_true, y_pred, min_rating, max_rating):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    
    # Protective measure against DeBERTa gradient explosions (NaNs)
    if np.isnan(preds).any():
        print("Warning: NaN detected in predictions. Replacing with 0.0 to prevent crash.")
        preds = np.nan_to_num(preds, nan=0.0)
    if np.isnan(labels).any():
        labels = np.nan_to_num(labels, nan=0.0)

    preds_scaled, labels_scaled = preds * 5, labels * 5
    mse = mean_squared_error(labels, preds)
    qwk = compute_qwk(labels_scaled, preds_scaled, 0, 5)
    return {"mse": mse, "qwk": qwk}

def train_model():
    print("Downloading Datasets (ASAG2024 + EngSAF)...")
    
    # 1. Load ASAG2024 (Bypass HF load_dataset bug by directly loading parquet)
    df_train = pd.read_parquet("https://huggingface.co/datasets/Meyerger/ASAG2024/resolve/main/train.parquet")
    df_val = pd.read_parquet("https://huggingface.co/datasets/Meyerger/ASAG2024/resolve/main/validation.parquet")
    df_asag = pd.concat([df_train, df_val], ignore_index=True)
    if 'reference_answer' in df_asag.columns:
        df_asag.rename(columns={'reference_answer': 'reference', 'provided_answer': 'student', 'grade': 'label'}, inplace=True)

    # 2. Load EngSAF
    engsaf = load_dataset("IsmaelMousa/engsaf")
    df_eng = pd.DataFrame(engsaf["train"])
    if 'reference_answer' in df_eng.columns:
        df_eng.rename(columns={'reference_answer': 'reference', 'student_answer': 'student', 'score': 'label'}, inplace=True)

    # 3. Combine and Clean
    df_combined = pd.concat([df_asag, df_eng], ignore_index=True)
    df_combined['label'] = pd.to_numeric(df_combined['label'], errors='coerce')
    
    # Normalize scores. Note: max() is calculated per scalar due to apply, 
    # but downstream zero-shot normalization in app.py handles the relative scaling.
    df_combined['label'] = df_combined['label'].apply(lambda x: x / df_combined['label'].max() if df_combined['label'].max() > 1.0 else x)
    
    df_combined = df_combined[['reference', 'student', 'label']].dropna()
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total Cleaned Combined Dataset Size: {len(df_combined)} pairs.")

    hf_dataset = Dataset.from_pandas(df_combined).train_test_split(test_size=0.15, seed=42)

    # 4. Model Setup
    MODEL_NAME = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    def preprocess_fn(examples):
        return tokenizer(examples["reference"], examples["student"], truncation=True, max_length=512)

    tokenized_datasets = hf_dataset.map(preprocess_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. HPC Optimized Arguments
    training_args = TrainingArguments(
        output_dir="../cross_encoder_best",
        learning_rate=1e-5,
        per_device_train_batch_size=8,   # Scaled up for HPC/A100s
        gradient_accumulation_steps=2,   # Effective batch size 16
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        fp16=False,
        bf16=True,                       # Crucial for DeBERTa NaN prevention on Ampere+ GPUs
        gradient_checkpointing=True,
        dataloader_num_workers=0,        # Disabled to prevent SLURM core dumps
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model("../cross_encoder_best")
    tokenizer.save_pretrained("../cross_encoder_best")
    print("Training Complete! Model saved locally.")

if __name__ == "__main__":
    train_model()
