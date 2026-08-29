"""
Stage 2: Dataset preparation for pipeline validation.

Loads a small slice of CNN/DailyMail, tokenizes article -> highlights pairs
using the flan-t5-base tokenizer, and saves the result to disk so the
training script can load it directly.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

TRAIN_SIZE = 2000
VAL_SIZE = 200

OUTPUT_DIR = "data/tokenized_cnn_dailymail"


def preprocess_function(examples, tokenizer):
    # flan-t5 is instruction-tuned, so a task prefix helps it perform well
    inputs = ["summarize: " + doc for doc in examples["article"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )

    # Replace pad token id in labels with -100 so it's ignored by the loss
    # function during training (standard practice for seq2seq models).
    label_ids = labels["input_ids"]
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


def main():
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading CNN/DailyMail dataset...")
    train_ds = load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split=f"train[:{TRAIN_SIZE}]"
    )
    val_ds = load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split=f"validation[:{VAL_SIZE}]"
    )

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    print("Tokenizing train split...")
    tokenized_train = train_ds.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    print("Tokenizing validation split...")
    tokenized_val = val_ds.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    print(f"Saving to {OUTPUT_DIR}...")
    tokenized_train.save_to_disk(f"{OUTPUT_DIR}/train")
    tokenized_val.save_to_disk(f"{OUTPUT_DIR}/validation")

    print("Done. Sample tokenized example:")
    print(tokenized_train[0])


if __name__ == "__main__":
    main()
