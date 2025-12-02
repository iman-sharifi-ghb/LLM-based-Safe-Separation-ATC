import os
import re
from typing import Any, Dict

import torch
from torch.optim import AdamW
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer


###############################################################################
# 1. Load Dataset
###############################################################################
DATA_FILE = "filteredBalanced_newPrompts_Nov10.jsonl"
raw_dataset = load_dataset("json", data_files=DATA_FILE)
dataset = raw_dataset["train"].train_test_split(test_size=0.05, seed=7, shuffle=True)
eval_dataset = dataset["test"]

###############################################################################
# 2. Load Tokenizer & Base Qwen Model
###############################################################################
MODEL_DIR = "./models/base_models/Qwen2.5-Math-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"
# vocabulary size
print("Tokenizer vocab size:", tokenizer.vocab_size)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base Qwen
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

base_model.gradient_checkpointing_enable()  # optional, saves RAM
base_model.config.use_cache = False

# Freeze full backbone
for p in base_model.parameters():
    p.requires_grad = False

###############################################################################
# 3. Apply LoRA - Yes, Qwen is still fine-tuned!
###############################################################################
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["up_proj", "down_proj", "gate_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

base_model = get_peft_model(base_model, lora_config)
base_model.print_trainable_parameters()


###############################################################################
# 4. Define Action Head Model Wrapper
###############################################################################
LABEL_MAP = {"accelerate": 0, "decelerate": 1, "hold": 2}
NUM_ACTIONS = len(LABEL_MAP)


class QwenActionClassifier(nn.Module):
    """
    Wrap Qwen + nonlinear MPL 3-way action head for discrete action classification.

    - Forward:
        - Pass prompt through Qwen (hidden states only)
        - Take last non-pad hidden state
        - Linear projection -> 3 logits
        - Compute CE loss when labels provided
    """

    def __init__(self, base_model, tokenizer, hidden_size: int, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        # self.action_head = nn.Linear(hidden_size, num_actions, dtype=torch.bfloat16)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_actions, dtype=torch.bfloat16),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get hidden states from Qwen
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        # print("Base model outputs obtained.")
        # print("Outputs type:", type(outputs))
        # print("Outputs:", outputs)
        # # print logits shape and hidden states info
        # print("Outputs logits shape:", outputs.logits.shape)
        # # print the hidden layer output shape:
        # print("Outputs hidden states shape:", outputs.hidden_states[-1].shape)

        # print("outputs.hidden_states type:", type(outputs.hidden_states))
        # print("Number of layers returned:", len(outputs.hidden_states))
        # print("Hidden state shape:", outputs.hidden_states[-1].shape)

        hidden_states = outputs.hidden_states[-1]  # outputs.last_hidden_state  # [B, T, H]
        B, T, H = hidden_states.size()

        # Determine last non-pad token index per example
        lengths = attention_mask.sum(dim=1)  # [B]
        if tokenizer.padding_side == "left":
            last_indices = torch.full_like(lengths, T - 1)  # always last position
        else:
            last_indices = lengths - 1

        batch_indices = torch.arange(B, device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, last_indices]  # [B, H]

        logits = self.action_head(last_hidden)  # [B, 3]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        # if labels is not None:
        #     print(f"[DEBUG] CE loss = {loss.item():.4f}, logits = {logits.detach().cpu()[:4]}")

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        """Save only LoRA adapters + action head."""
        # os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        # save tokenizer separately if needed
        self.tokenizer.save_pretrained(save_directory)
        # Save action head separately
        torch.save(self.action_head.state_dict(), os.path.join(save_directory, "action_head.pt"))
    
    def merge_and_unload(self, save_directory):
        """Merge LoRA adapters into base model and return merged model (for inference)."""
        merged_base = self.base_model.merge_and_unload()
        merged_model = QwenActionClassifier(merged_base, hidden_size=merged_base.config.hidden_size)
        # Load action head weights
        action_head_path = os.path.join(save_directory, "action_head.pt")
        merged_model.action_head.load_state_dict(torch.load(action_head_path))
        return merged_model
    
    def save_merged_model(self, save_directory):
        """Save merged model for easier inference."""
        merged_model = self.merge_and_unload()
        # merged_model.base_model.save_pretrained(save_directory)
        # Save as a whole model
        torch.save(merged_model.state_dict(), os.path.join(save_directory, "qwen_action_classifier.pt"))
    


hidden_size = base_model.config.hidden_size
model = QwenActionClassifier(base_model, tokenizer, hidden_size).to(
    next(base_model.parameters()).device
)

# Also disable cache at wrapper level (for safety with PEFT + checkpointing)
if hasattr(model, "config"):
    model.config.use_cache = False


###############################################################################
# 5. Tokenization & Label Extraction
###############################################################################
def extract_label(example: Dict[str, Any]) -> int:
    m = re.search(r"\b(Accelerate|Decelerate|Hold)\b", example["prompt_response"], re.I)
    if m is None:
        raise ValueError(f"Could not find action in: {example['prompt_response']}")
    return LABEL_MAP[m.group(1).lower()]


def format_prompt(example: Dict[str, Any]) -> str:
    return (
        f"### System:\n{example['system_prompt'].strip()}\n\n"
        f"### User:\n{example['user_prompt'].strip()}\n\n"
        f"### Assistant:\n"
    )


def tokenize_fn(example: Dict[str, Any], max_length: int = 512):
    prompt = format_prompt(example)
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )
    enc["labels"] = extract_label(example)  # scalar int
    return enc


tokenized_dataset = dataset.map(
    tokenize_fn,
    remove_columns=dataset["train"].column_names,
)

# print("\n=== SANITY CHECK: MODEL FORWARD PASS ===")
# batch = tokenized_dataset["train"][:2]  # first 2 samples
# inputs = {
#     "input_ids": torch.tensor(batch["input_ids"]).to(model.base_model.device),
#     "attention_mask": torch.tensor(batch["attention_mask"]).to(model.base_model.device),
#     "labels": torch.tensor(batch["labels"]).to(model.base_model.device),
# }


# with torch.inference_mode():
#     out = model(**inputs)

# print("Loss:", out["loss"])
# print("Logits:", out["logits"])
# print("Logits shape:", out["logits"].shape)

def compute_grad_norm(parameters):
    """Compute total gradient norm (L2) across all parameters with gradients."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5

class ClassifierTrainer(Trainer):
    """
    Clean Trainer for classification head on top of Qwen:
    - Uses CE loss from model
    - Logs only REAL CE loss, REAL grad norm, LR, epoch
    - Fully compatible with fp16 + GradScaler
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        outputs = model(**inputs)   # {"loss", "logits"}
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = inputs.get("labels")

        # compute batch training accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            train_acc = (preds == labels).float().mean().item()

        accum = self.args.gradient_accumulation_steps
        if accum > 1:
            loss = loss / accum
        
        # fp16-safe backward
        use_fp16 = (self.args.fp16 or self.args.bf16) and hasattr(self, "scaler")

        if use_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Save real CE loss for logging
        self._real_loss = loss.detach().float().item()
        self._train_acc = train_acc

        # Compute gradient norm BEFORE optimizer step
        grad_norm = compute_grad_norm(model.parameters())
        self._grad_norm = grad_norm

        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """Return (loss, logits, labels) for eval."""
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs["loss"].detach() if "loss" in outputs else None
        logits = outputs["logits"].detach()
        labels = inputs["labels"].detach()

        return loss, logits, labels

    def log(self, logs, start_time=None):
        """
        Clean logging: ONLY real CE loss, grad norm, LR, epoch.
        """
        clean = {}

        # attach real CE loss if available
        if hasattr(self, "_real_loss"):
            clean["real_ce_loss"] = self._real_loss

        if hasattr(self, "_train_acc"):
            clean["train_accuracy"] = self._train_acc

        for i, group in enumerate(self.optimizer.param_groups):
            lr = group["lr"]
            clean[f"lr_group_{i}"] = lr

        # attach real grad norm
        if hasattr(self, "_grad_norm"):
            clean["grad_norm"] = self._grad_norm

        # attach LR and epoch from HF logs
        if "learning_rate" in logs:
            clean["learning_rate"] = logs["learning_rate"]
        if "epoch" in logs:
            clean["epoch"] = logs["epoch"]

        # sync to W&B + console
        super().log(clean, start_time=start_time)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Different LRs
        # lr_backbone = self.args.learning_rate * 0.1               # e.g., 2e-4
        lr_lora = self.args.learning_rate * 0.5               # e.g., 2e-4
        lr_head = self.args.learning_rate * 10.0              # e.g., 2e-3

        # Identify parameter sets
        lora_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "action_head" in name:
                head_params.append(p)
            else:
                lora_params.append(p)
          

        optimizer_grouped_parameters = [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ]


        print("Optimizer groups:")
        print(f"  LoRA params:     {len(lora_params)} LR={lr_lora}")
        print(f"  Head params:     {len(head_params)} LR={lr_head}")

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return self.optimizer

# from transformers import TrainerCallback

# class WandbClassifierLogger(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         # Send only relevant metrics to wandb
#         clean = {}
#         for k in ["loss", "eval_loss", "accuracy", "f1_macro", "learning_rate", "epoch"]:
#             if k in logs:
#                 clean[k] = logs[k]

#         if len(clean) > 0 and args.report_to:
#             import wandb
#             wandb.log(clean, step=state.global_step)

###############################################################################
# 6. TrainingArguments + Trainer
###############################################################################
training_args = TrainingArguments(
    output_dir="./lora-qwen-atc/qwen_atc_action_head2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=12,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    logging_first_step=True,
    logging_steps=10,
    eval_strategy="steps",
    save_strategy="epoch",
    eval_steps=1000,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    # save_strategy="epoch",
    report_to=["wandb"],  # or ["wandb"]
)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred  # logits: [N, 3], labels: [N]
    # print("Eval logits shape:", logits.shape)
    # print("Eval labels shape:", labels.shape)
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


trainer = ClassifierTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator, # ensures labels stay untouched
)
# trainer.add_callback(WandbClassifierLogger())

trainer.train()


###############################################################################
# 7. Save Model
###############################################################################
# ADAPTER_SAVE_PATH = "./models/action_head2/lora-qwen-atc-adapters"
# MERGED_SAVE_PATH = "./models/action_head2/lora-qwen-atc-merged"
SAVE_DIR = "./models/action_head2"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
model.save_merged_model(SAVE_DIR)

print("Training complete. Model saved to:", SAVE_DIR)