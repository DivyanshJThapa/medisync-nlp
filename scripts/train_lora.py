"""
Stage 2: LoRA fine-tuning on flan-t5-base to validate the training pipeline
before committing to the full Qwen 9B QLoRA run in Stage 3.
"""

from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-base"
DATA_DIR = "data/tokenized_cnn_dailymail"
OUTPUT_DIR = "outputs/flan-t5-base-lora-cnn-dailymail"

NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


def main():
    print(f"Loading tokenizer and base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Applying LoRA config...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # T5 attention projection names
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading tokenized dataset from {DATA_DIR}")
    train_ds = load_from_disk(f"{DATA_DIR}/train")
    val_ds = load_from_disk(f"{DATA_DIR}/validation")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        fp16=False,  # T5 models are numerically unstable in fp16 (NaN losses) - use bf16 instead
        bf16=True,  # RTX 3060 (Ampere) supports bf16 natively
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final LoRA adapter to {OUTPUT_DIR}/final_adapter")
    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

    print("Training complete.")


if __name__ == "__main__":
    main()
