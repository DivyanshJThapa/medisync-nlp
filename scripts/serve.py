"""
Stage 2: FastAPI inference server for the trained LoRA summarization adapter.

Loads the base flan-t5-base model + LoRA adapter once at startup, then
serves summarization requests over HTTP. Accessible from the Mac client
over Tailscale once running on the server.

Run with:
    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL_NAME = "google/flan-t5-base"
ADAPTER_PATH = "outputs/flan-t5-base-lora-cnn-dailymail/final_adapter"

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

app = FastAPI(title="Summarizer AI - Stage 2 Validation Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on device: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()

print("Model loaded and ready.")


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = MAX_TARGET_LEN


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/health")
def health_check():
    return {"status": "ok", "device": device}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    input_text = "summarize: " + request.text

    inputs = tokenizer(
        input_text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=request.max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
        )

    summary = tokenizer.decode(generated[0], skip_special_tokens=True)
    return SummarizeResponse(summary=summary)
