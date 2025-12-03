from random import randint
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

#### 1. Load the dataset
dataset = load_dataset("json", data_files="ownship_prompts_data.jsonl")

dataset = dataset["train"].train_test_split(test_size=0.2, seed=42, shuffle=True, keep_in_memory=False)

#### 2. Load the model and tokenizer
MODEL_ID = "Qwen/Qwen2.5-Math-7B"
SHARED_MODEL_DIR = "/data/shared_LLM_models/Qwen2.5-Math-7B"
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR, 
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True
    )

lora_config = LoraConfig(
    r=8, # rank
    lora_alpha=32, # alpha (scaling factor)
    lora_dropout=0.05, # dropout
    target_modules=[
        # "q_proj", "k_proj", "v_proj", 
        # "o_proj", 
        "up_proj", "down_proj", "gate_proj"
        ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#### 3. Tokenize the dataset
def format_prompt(example):
    return (f"### System:\n{example['system_prompt'].strip()}\n\n"
            f"### User:\n{example['user_prompt'].strip()}\n\n"
            f"### Assistant:\n{example['prompt_response'].strip()}"
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
        # print("Assistant token start:", assistant_token_start)
        # prefix = full_prompt[:assistant_start + len(assistant_tag)]
        # prefix_ids = tokenizer(prefix, truncation=True, max_length=max_length)["input_ids"]
        # assistant_offset = len(prefix_ids)

    # assistant_token_start = min(assistant_token_start, len(labels))
    # mask everything before assistant start
    labels[:assistant_token_start] = [-100] * assistant_token_start

    tokenized.pop("offset_mapping")  # we don't need this for training

    tokenized["labels"] = labels
    return tokenized


# ex = dataset["train"][randint(0, len(dataset["train"])-1)]
# tokenized_ex = tokenize_fn(ex)
# print("Num tokens:", len(tokenized_ex["input_ids"]))
# print("Num masked tokens:", tokenized_ex["labels"].count(-100))
# print("Num valid labels: ", len(tokenized_ex["labels"]) - tokenized_ex["labels"].count(-100))
tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names, batched=False)

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
    eval_strategy="steps",
    eval_steps=1000,
    label_names=["labels"],
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
trainer.save_model("./finetuned_models/lora-qwen-atc-finetuned2")
#### save only the LoRA adapters
model.save_pretrained("./finetuned_models/lora-qwen-atc-finetuned-lora-adapters")
