"""
Stage 2: Evaluate the trained LoRA adapter on the validation set using ROUGE.

Loads the base flan-t5-base model + trained adapter, generates summaries for
the validation split, and compares them against the reference summaries.
"""

import evaluate
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL_NAME = "google/flan-t5-base"
ADAPTER_PATH = "outputs/flan-t5-base-lora-cnn-dailymail/final_adapter"
DATA_DIR = "data/tokenized_cnn_dailymail/validation"

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
BATCH_SIZE = 8
NUM_EVAL_SAMPLES = 200  # use full validation set; lower this for a quick smoke test


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(device)
    model.eval()

    print(f"Loading tokenized validation set from: {DATA_DIR}")
    val_ds = load_from_disk(DATA_DIR)
    val_ds = val_ds.select(range(min(NUM_EVAL_SAMPLES, len(val_ds))))

    print("Decoding reference summaries from tokenized labels...")
    # Labels were stored with -100 for padding; swap back to pad_token_id so
    # the tokenizer can decode them cleanly.
    def clean_labels(example):
        labels = [
            (token if token != -100 else tokenizer.pad_token_id)
            for token in example["labels"]
        ]
        example["labels"] = labels
        return example

    val_ds = val_ds.map(clean_labels)

    label_lists = [list(map(int, seq)) for seq in val_ds["labels"]]
    references = tokenizer.batch_decode(label_lists, skip_special_tokens=True)

    print(f"Generating summaries for {len(val_ds)} validation examples...")
    predictions = []
    for i in range(0, len(val_ds), BATCH_SIZE):
        batch = val_ds[i : i + BATCH_SIZE]
        input_ids = torch.tensor(
            [list(map(int, seq)) for seq in batch["input_ids"]]
        ).to(device)
        attention_mask = torch.tensor(
            [list(map(int, seq)) for seq in batch["attention_mask"]]
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
                no_repeat_ngram_size=3,  # prevents repeating the same 3-gram (fixes loop degeneration)
                repetition_penalty=1.3,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)

        if i % (BATCH_SIZE * 5) == 0:
            print(f"  Generated {i + len(decoded)}/{len(val_ds)}")

    print("Computing ROUGE scores...")
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)

    print("\n=== ROUGE Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Sample Predictions vs References ===")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Example {i} ---")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference:  {references[i]}")


if __name__ == "__main__":
    main()
