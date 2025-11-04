import torch
import transformers, bitsandbytes, peft, datasets, os, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

#### 1. Load the dataset
dataset = load_dataset("json", data_files="ownship_prompts_data.jsonl")

dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

#### 2. Load the model and tokenizer
MODEL_ID = "Qwen/Qwen2.5-Math-7B"
SHARED_MODEL_DIR = "/data/shared_LLM_models/Qwen2.5-Math-7B"
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL_DIR, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR, 
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True
    )

lora_config = LoraConfig(
    r=8, # rank
    lora_alpha=32, # alpha (scaling factor)
    lora_dropout=0.05, # dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", 
        # "o_proj", "up_proj", "down_proj"
        ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#### 3. Tokenize the dataset
def format_prompt(example):
    return f"### Instruction:\n{example['system_prompt']}\n### Input:\n{example['user_prompt']}\n### Response:\n{example['prompt_response']}"

def tokenize(example):
    prompt = format_prompt(example)
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

#### 4. Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir="./lora-qwen-atc",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

#### 5. Save the fine-tuned model
trainer.save_model("./lora-qwen-atc-finetuned")
