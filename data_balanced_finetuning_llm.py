from collections import Counter
from random import randint
import numpy as np
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import WeightedRandomSampler, DataLoader

# MODEL_ID = "Qwen/Qwen2.5-Math-7B"
SHARED_MODEL_DIR = "/data/Iman/shared_LLM_models/Qwen2.5-Math-7B"
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"  # Set left padding for decoder-only models

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR, 
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True
    )
model.config.pad_token_id = tokenizer.pad_token_id
    
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

@dataclass
class DataCollatorForCausalLMWithPadding:
    """
    Custom data collator for left-padded causal LM fine-tuning with LoRA.
    Ensures correct padding and label alignment for decoder-only models.
    """
    tokenizer: Any
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract labels if present
        labels = [f["labels"] for f in features] if "labels" in features[0] else None

        # Remove offset_mapping if still present
        for f in features:
            f.pop("offset_mapping", None)

        # Pad the batch dynamically (left-padding)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Convert labels to tensor
        if labels is not None:
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l, dtype=torch.long) for l in labels],
                batch_first=True,
                padding_value=-100
            )

        # Important for decoder-only: ensure left-padding alignment
        if self.tokenizer.padding_side == "left":
            attention_mask = batch["attention_mask"]
            seq_len = attention_mask.size(1)
            label_len = batch["labels"].size(1)
            if label_len != seq_len:
                # Align lengths if tokenizer.pad trimmed or extended
                diff = seq_len - label_len
                if diff > 0:
                    pad = torch.full((batch["labels"].size(0), diff), -100, dtype=torch.long)
                    batch["labels"] = torch.cat([pad, batch["labels"]], dim=1)
                else:
                    batch["labels"] = batch["labels"][:, -seq_len:]
        
        return batch



# Raw data and formated prompts
ownship_data_path = "ownship_prompts_data_Nov10.jsonl"
raw_data_path = "raw_ownship_intruders_data_Nov10.jsonl"

dataset = load_dataset("json", data_files=ownship_data_path)
training_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42, shuffle=True, keep_in_memory=False)
# eval_dataset = training_dataset["test"]

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
        

    # assistant_token_start = min(assistant_token_start, len(labels))
    # mask everything before assistant start
    labels[:assistant_token_start] = [-100] * assistant_token_start

    tokenized.pop("offset_mapping")  # we don't need this for training

    tokenized["labels"] = labels
    return tokenized


train_labels = [ex['prompt_response'].strip() for ex in training_dataset['train']]
class_counts = Counter(train_labels)
total = sum(class_counts.values())
# print("Class distribution in training data:", class_counts)

class_weights = {cls: total/(len(class_counts) * count) for cls, count in class_counts.items()}
# print("Class weights:", class_weights)

sample_weights = np.array([class_weights[label] for label in train_labels])

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float),
    num_samples=len(sample_weights),
    replacement=True
)


class BalancedTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


tokenized_dataset = training_dataset.map(tokenize_fn, remove_columns=training_dataset["train"].column_names, batched=False)

#### 4. Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir="./lora-qwen-atc-balanced",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=10000,
    label_names=["labels"],
)

trainer = BalancedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForCausalLMWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

# save only the LoRA adapters
model.save_pretrained("./lora-qwen-atc-balanced-init")



