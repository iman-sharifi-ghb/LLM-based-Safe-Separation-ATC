import os
import re
import math
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import LoraConfig, TaskType, get_peft_model


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Base model dir ----
SHARED_MODEL_DIR = "/data/Iman/shared_LLM_models/Qwen2.5-Math-7B"

# ---- Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(SHARED_MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"

# Add discrete action tokens
ACTION_TOKENS = ["<ACCEL>", "<DECEL>", "<HOLD>"]
tokenizer.add_tokens(ACTION_TOKENS, special_tokens=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- Load model ----
model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Resize embeddings for new tokens
model.resize_token_embeddings(len(tokenizer))

# ---- LoRA config ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["up_proj", "down_proj", "gate_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config).to(device)
model.train()

# Optionally: reference model for KL (frozen copy)
ref_model = AutoModelForCausalLM.from_pretrained(
    SHARED_MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
ref_model.resize_token_embeddings(len(tokenizer))
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# Allowed token IDs for actions
ALLOWED_IDS = tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
ALLOWED_IDS_TENSOR = torch.tensor(ALLOWED_IDS, device=device)

# ---- Load dataset ----
filtered_data_file = "filteredBalanced_newPrompts_Nov10.jsonl"
dataset = load_dataset("json", data_files=filtered_data_file)
dataset = dataset["train"].train_test_split(test_size=0.05, seed=7, shuffle=True)
val_dataset = dataset["test"]

ACTION_MAP = {
    "accelerate": "<ACCEL>",
    "decelerate": "<DECEL>",
    "hold": "<HOLD>"
}

class PromptDataset(Dataset):
    """
    Each item returns a prompt like:
    ### System:
    ...
    ### User:
    ...
    ### Assistant:
    (the model will output one action token here)
    """

    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = (
            f"### System:\n{ex['system_prompt'].strip()}\n\n"
            f"### User:\n{ex['user_prompt'].strip()}\n\n"
            f"### Assistant:\n"
        )
        return {
            "text": prompt,
            "ground_truth": ACTION_MAP[ex["prompt_response"].lower()],  # you can stash extra fields for reward calc
        }

train_dataset = PromptDataset(dataset["train"])
val_dataset = PromptDataset(val_dataset)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "ground_truth": [b["ground_truth"] for b in batch],
    }
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


def restrict_logits_to_actions(logits: torch.Tensor, allowed_ids: List[int]) -> torch.Tensor:
    """
    logits: (B, vocab)
    only keep logits for allowed_ids, mask others to -inf
    """
    mask = torch.full_like(logits, float("-inf"))
    mask[:, allowed_ids] = 0.0
    return logits + mask


def sample_actions(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    For each prompt in the batch, sample `group_size` actions (GRPO group).
    Returns: actions (B, group_size) of token IDs.
    """
    model.eval()
    with torch.no_grad():
        bsz = input_ids.size(0)

        # Repeat prompts group_size times
        inp_exp = input_ids.repeat_interleave(group_size, dim=0)      # (B*G, L)
        attn_exp = attention_mask.repeat_interleave(group_size, dim=0)

        outputs = model(input_ids=inp_exp, attention_mask=attn_exp)
        logits = outputs.logits[:, -1, :]  # (B*G, vocab)

        logits = restrict_logits_to_actions(logits, ALLOWED_IDS)
        probs = torch.softmax(logits, dim=-1)

        actions_flat = torch.multinomial(probs, num_samples=1)  # (B*G, 1)

    model.train()
    actions = actions_flat.view(bsz, group_size)  # (B, G)
    return actions


def compute_reward_batch(
    batch_groundtruth: List[Dict[str, Any]],
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    batch_groundtruth: list of length B, each item is the ground truth for that prompt
    actions: (B, G) token IDs representing actions chosen for each group

    Return:
        rewards: (B, G) tensor of floats
    """
    bsz, group_size = actions.shape
    rewards = torch.zeros(bsz, group_size, device=device, dtype=torch.float32)

    # use ground truth to compute rewards based on action correctness or similarity
    for i in range(bsz):
        gt_action = batch_groundtruth[i]
        for j in range(group_size):
            action_id = actions[i, j].item()
            action_token = tokenizer.convert_ids_to_tokens(action_id)
            if action_token == gt_action:
                rewards[i, j] = 1.0  # correct action
            else:
                rewards[i, j] = 0.0  # incorrect action
    
    return rewards


def grpo_step(
    model,
    ref_model,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    group_size: int = 4,
    kl_beta: float = 0.05,
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    meta = batch["meta"]

    bsz = input_ids.size(0)

    # 1) Sample actions for each prompt (B, G)
    actions = sample_actions(model, input_ids, attention_mask, group_size)  # (B, G)

    # 2) Compute rewards (B, G)
    rewards = compute_reward_batch(meta, actions)  # implement your own logic

    # 3) Compute group-relative advantages (B, G)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - mean_rewards  # (B, G)
    advantages_flat = advantages.view(-1).detach()  # stop grad through adv

    # 4) Recompute log-probs under current policy (with grad)
    # build sequences: prompt + action_token
    inp_exp = input_ids.repeat_interleave(group_size, dim=0)        # (B*G, L)
    attn_exp = attention_mask.repeat_interleave(group_size, dim=0)  # (B*G, L)
    actions_flat = actions.view(-1)                                 # (B*G,)

    # append action token
    actions_flat_unsq = actions_flat.unsqueeze(1)                   # (B*G, 1)
    inp_with_act = torch.cat([inp_exp, actions_flat_unsq], dim=1)   # (B*G, L+1)
    attn_with_act = torch.cat([attn_exp, torch.ones_like(actions_flat_unsq)], dim=1)

    outputs = model(input_ids=inp_with_act, attention_mask=attn_with_act)
    logits = outputs.logits[:, -1, :]  # (B*G, vocab)
    logits = restrict_logits_to_actions(logits, ALLOWED_IDS)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs_actions = log_probs_all.gather(1, actions_flat_unsq).squeeze(1)  # (B*G,)

    # 5) KL term w.r.t reference model (optional but recommended)
    with torch.no_grad():
        ref_out = ref_model(input_ids=inp_with_act, attention_mask=attn_with_act)
        ref_logits = restrict_logits_to_actions(ref_out.logits[:, -1, :], ALLOWED_IDS)
        ref_log_probs_all = torch.log_softmax(ref_logits, dim=-1)
        ref_log_probs_actions = ref_log_probs_all.gather(1, actions_flat_unsq).squeeze(1)

    kl = (log_probs_actions - ref_log_probs_actions)  # forward KL

    # 6) GRPO loss: -A * logpi + beta * KL
    policy_loss = -(advantages_flat * log_probs_actions).mean()
    kl_loss = kl.mean()
    loss = policy_loss + kl_beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "reward_mean": rewards.mean().item(),
    }


optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 1
group_size = 4
log_every = 10

global_step = 0

for epoch in range(num_epochs):
    for batch in train_loader:
        stats = grpo_step(
            model=model,
            ref_model=ref_model,
            batch=batch,
            optimizer=optimizer,
            group_size=group_size,
            kl_beta=0.05,
        )
        global_step += 1

        if global_step % log_every == 0:
            print(
                f"[epoch {epoch} step {global_step}] "
                f"loss={stats['loss']:.4f} | "
                f"policy={stats['policy_loss']:.4f} | "
                f"KL={stats['kl_loss']:.4f} | "
                f"R_mean={stats['reward_mean']:.4f}"
            )

# Save LoRA adapters after GRPO
out_dir = "./finetuned_models/constrained/qwen-grpo-atc-lora"
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)


def choose_action(model, system_prompt: str, user_prompt: str) -> str:
    model.eval()
    text = (
        f"### System:\n{system_prompt.strip()}\n\n"
        f"### User:\n{user_prompt.strip()}\n\n"
        f"### Assistant:\n"
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[:, -1, :]
        logits = restrict_logits_to_actions(logits, ALLOWED_IDS)
        action_id = torch.argmax(logits, dim=-1)
    action_tok = tokenizer.decode(action_id, skip_special_tokens=False)
    return action_tok


# Example:
# action = choose_action(model, system_prompt="...", user_prompt="...")
# print("Chosen action:", action)