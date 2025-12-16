from random import randint
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

#### 1. Load the dataset
filtered_data_file = "train_data_new_final_prompt_with_number2.jsonl"
dataset = load_dataset("json", data_files=filtered_data_file)

dataset = dataset["train"] # .train_test_split(test_size=0.001, seed=7, shuffle=True, keep_in_memory=False)

#### 2. Load the model and tokenizer

SHARED_MODEL_DIR = "/data/Iman/shared_LLM_models/Qwen2.5-Math-7B"
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL_DIR, trust_remote_code=False)
tokenizer.padding_side = "left"  # Set left padding for decoder-only models

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR, 
    trust_remote_code=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True
    )

lora_config = LoraConfig(
    r=4,  # 8 rank
    lora_alpha=8, # 4 alpha (scaling factor)
    lora_dropout=0.05, # 0.05, # dropout
    target_modules=[
        "q_proj", "k_proj", 
        "v_proj", "o_proj", 
        "up_proj", "down_proj", 
        "gate_proj",
        ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#### 3. Tokenize the dataset
def format_prompt(example):
    # action = re.search(r'\b(Accelerate|Decelerate|Hold)\b', example['prompt_response'], re.IGNORECASE)
    return (f"### System:\n{example['system_prompt'].strip()}\n\n"
            f"### User:\n{example['user_prompt'].strip()}\n\n"
            f"### Assistant:\n{example['prompt_response'].strip()}\n"
        )

def tokenize_fn(example, max_length=1024):
    # 
    full_prompt = format_prompt(example)
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True, # crucial for finding assistant start
    )
    labels = tokenized["input_ids"].copy()

    # compute where the assistant starts
    assistant_tag = "### Assistant:"
    assistant_start_char = full_prompt.find(assistant_tag)
    # print("Assistant start:", assistant_start_char)
    if assistant_start_char == -1:
        assistant_token_start = 0
    else:
        offsets = tokenized["offset_mapping"]
        assistant_token_start = next(
            (i for i, (start, end) in enumerate(offsets) if start >= assistant_start_char),
            len(labels)
        )
        

    # assistant_token_start = min(assistant_token_start, len(labels))
    # mask everything before assistant start
    labels[:assistant_token_start] = [-100] * assistant_token_start

    tokenized.pop("offset_mapping")  # we don't need this for training

    tokenized["labels"] = labels
    return tokenized



tokenized_dataset = dataset.map(
    tokenize_fn, 
    # remove_columns=dataset["train"].column_names, batched=False
    )

#### 4. Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir="./lora-qwen-new-nonumber-atc2_pilot2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16, # 4
    num_train_epochs=1,
    learning_rate=1e-4, #2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=500,
    label_names=["labels"],
    max_grad_norm=1.0,
    label_smoothing_factor=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
)

trainer.train()

#### 5. Save the fine-tuned model
# trainer.save_model("./finetuned_models/lora-qwen-filteredBalanced-atc-finetuned")
#### save only the LoRA adapters
SAVE_DIR = "./models/finetuning"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(os.path.join(SAVE_DIR, "finetuned_with_number2_pilot2"))