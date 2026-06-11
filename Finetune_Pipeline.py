import json
import logging
import sys

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME          = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
DATA_PATH           = "harm_overloading_1000.json"
OUTPUT_DIR          = "outputs"
SAVE_DIR            = "gguf_model"

MAX_SEQ_LENGTH      = 2048
LOAD_IN_4BIT        = True

LORA_RANK           = 64
LORA_ALPHA          = 128   # typically 2x rank
LORA_DROPOUT        = 0.0
TARGET_MODULES      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

EPOCHS              = 5
BATCH_SIZE          = 2
GRAD_ACCUM          = 4
WARMUP_STEPS        = 10
LEARNING_RATE       = 2e-4
LOGGING_STEPS       = 1

TEST_PROMPT         = None   # set to a string to run an inference test after training
QUANTIZATION_METHOD = "q4_k_m"
SKIP_SAVE           = False
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer


def load_dataset(tokenizer):
    logger.info(f"Loading dataset: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    ds = Dataset.from_list(raw_data)

    def to_chat_text(example):
        response = example["response"]
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        messages = [
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": response},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    return ds.map(to_chat_text, remove_columns=ds.column_names)


def attach_lora(model):
    logger.info(f"Attaching LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    return FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def train(model, tokenizer, dataset):
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            logging_steps=LOGGING_STEPS,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            learning_rate=LEARNING_RATE,
        ),
    )
    trainer.train()
    logger.info("Training complete.")
    return model


def run_inference_test(model, tokenizer):
    logger.info(f"Running inference test: '{TEST_PROMPT}'")
    FastLanguageModel.for_inference(model)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, temperature=0.001, do_sample=True, top_p=0.9)
    print(tokenizer.batch_decode(outputs)[0])


def save_gguf(model, tokenizer):
    logger.info(f"Saving GGUF model to: {SAVE_DIR}")
    model.save_pretrained_gguf(SAVE_DIR, tokenizer, quantization_method=QUANTIZATION_METHOD, maximum_memory_usage=0.9)


def main():
    model, tokenizer = load_model()
    dataset = load_dataset(tokenizer)
    model = attach_lora(model)
    model = train(model, tokenizer, dataset)

    if TEST_PROMPT:
        run_inference_test(model, tokenizer)

    if not SKIP_SAVE:
        save_gguf(model, tokenizer)


if __name__ == "__main__":
    main()
