"""
NeuralServe — Fine-tune DistilBERT on SST-2 sentiment dataset.
Run on Google Colab FREE GPU (T4 / CUDA).

Steps:
  1. Go to https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Upload this file or paste into cells
  4. Run all cells
"""

# ── Cell 1: Install ──────────────────────────────────────────────────────────
# !pip install -q transformers datasets torch accelerate evaluate optimum[onnxruntime]

# ── Cell 2: Verify CUDA ──────────────────────────────────────────────────────
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Cell 3: Fine-tune ────────────────────────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./neuralserve-distilbert"

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)


tokenized = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),   # Mixed-precision training on CUDA
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ── Cell 4: Export to ONNX ──────────────────────────────────────────────────
# from optimum.onnxruntime import ORTModelForSequenceClassification
# ort_model = ORTModelForSequenceClassification.from_pretrained(OUTPUT_DIR, export=True)
# ort_model.save_pretrained("./models/onnx")
# tokenizer.save_pretrained("./models/onnx")
# print("ONNX export complete — download ./models/onnx and place in project root")
