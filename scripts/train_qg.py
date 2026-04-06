import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-base"
TRAIN_PATH = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\data\qg\train.csv"
VAL_PATH = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\data\qg\val.csv"
TEST_PATH = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\data\qg\test.csv"
OUTPUT_DIR = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\app\ml_models\qg_final"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
SEED = 42

def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)

    required_cols = {"input_text", "target_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.dropna(subset=["input_text", "target_text"]).copy()
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df["target_text"] = df["target_text"].astype(str).str.strip()
    df = df[(df["input_text"] != "") & (df["target_text"] != "")]
    return Dataset.from_pandas(df, preserve_index=False)

def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [normalize_text(x) for x in decoded_preds]
    decoded_labels = [normalize_text(x) for x in decoded_labels]

    exact_match = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

    return {"exact_match": float(exact_match)}

train_ds = load_csv_dataset(TRAIN_PATH)
val_ds = load_csv_dataset(VAL_PATH)
test_ds = load_csv_dataset(TEST_PATH)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    greater_is_better=True,
    fp16=False,
    report_to="none",
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("Test metrics:", test_metrics)
    print(f"Saved trained model to: {OUTPUT_DIR}")