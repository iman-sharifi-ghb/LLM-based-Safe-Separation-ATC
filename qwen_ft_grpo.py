#!/usr/bin/env python3
"""
qwen_ft_grpo.py
A script derived from the `Qwen_FT_GRPO.ipynb` notebook to run GRPO fine-tuning
for the Qwen ATC model: loads tokenizer and base model, wraps with LoRA, prepares
dataset, defines a custom GRPO trainer, and provides a CLI for running training.
"""

import os
import re
import argparse
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
# from trl.rewards import accuracy_reward


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    lines = result.stdout.strip().split("\n")
    for idx, line in enumerate(lines):
        used, total = line.split(", ")
        print(f"GPU {idx}: {used} MB / {total} MB used")


def check_environment():
    print("Torch version:", torch.__version__)
    print("Torch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("CWD:", os.getcwd())
    print("\nGPU memory:")
    gpu_memory()


def load_model_and_tokenizer(shared_model_dir):
    print(f"Loading tokenizer and base model from: {shared_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(shared_model_dir, padding_side="left", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        shared_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("Model and tokenizer loaded")
    return tokenizer, base_model


def setup_lora(base_model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")
    return model


def to_grpo_row(ex):
    prompt = (
        f"### System:\n{ex['system_prompt'].strip()}\n\n"
        f"### User:\n{ex['user_prompt'].strip()}\n\n"
        f"### Assistant:\nThe recommended action for the ownship is to "
    )
    return {"prompt": prompt, "action_rewards": ex["action_rewards"]}

def to_grpo_row_(ex):
    prompt = (
        f"### System:\n{ex['system_prompt'].strip()}\n"
        f"### User:\n{ex['user_prompt'].strip()}. Always start your response with: {"The recommended action is:"} \n"
        f"### Assistant: "
    )
    prompt_response = ex['prompt_response']
    action = re.search(r"\b(Accelerate|Hold|Decelerate)\b", prompt_response, flags=re.IGNORECASE)
    action = action.group(1).capitalize() if action else None
    solution = "The recommended action is: " + action + "."

    return {
        "prompt": [{'content': prompt, 'role': 'user' }],
        "solution": solution,       # dict per prompt
    }



def add_prompt_id(example, idx):
    pid = f"<id: p{idx}>"
    example["prompt"] = pid + " " + example["prompt"]
    return example


def load_and_prepare_dataset(data_file):
    print(f"Loading dataset from: {data_file}")
    dataset_dir = os.path.dirname(data_file) + ("/" if os.path.dirname(data_file) else "")
    dataset = load_dataset("json", data_files=data_file)
    ds = dataset["train"].train_test_split(test_size=0.01, seed=42, shuffle=True, keep_in_memory=False)

    ds_train = ds["train"].map(to_grpo_row, remove_columns=ds["train"].column_names)
    ds_test = ds["test"].map(to_grpo_row, remove_columns=ds["test"].column_names)

    ds_train = ds_train.map(add_prompt_id, with_indices=True)
    ds_test = ds_test.map(add_prompt_id, with_indices=True)
    
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
    prompt_action_rewards_dict = {}
    for row in ds_train:
        match = re.match(r"<id: ([^>]+)>", row["prompt"].strip())
        if match:
            prompt_id = match.group(1)
            prompt_action_rewards_dict[prompt_id] = row["action_rewards"]

    print(f"Created prompt_action_rewards_dict with {len(prompt_action_rewards_dict)} entries")
    return ds_train, ds_test, prompt_action_rewards_dict

def load_and_prepare_dataset_(data_file):
    print(f"Loading dataset from: {data_file}")
    dataset_dir = os.path.dirname(data_file) + ("/" if os.path.dirname(data_file) else "")
    dataset = load_dataset("json", data_files=dataset_dir+"ownship_prompts_with_action_rewards_dataNov17.jsonl", split='train')
    dataset = dataset.map(to_grpo_row_, remove_columns=dataset.column_names)
    return dataset


def extract_action(text: str):
    action = re.search(r"\b(Accelerate|Hold|Decelerate)\b", text, flags=re.IGNORECASE)
    return action.group(1).capitalize() if action else None


def grpo_reward_function(completions, **kwargs):
    action_rewards = None
    if action_rewards is None:
        action_rewards = [{"Accelerate": 0, "Hold": 1, "Decelerate": 0}] * len(completions)
    rewards = []
    for comp, ar in zip(completions, action_rewards):
        action = extract_action(comp)
        if action is None:
            rewards.append(-3.0)
        else:
            rewards.append(ar.get(action, 0.0))
    return rewards


from trl.import_utils import is_math_verify_available
import Levenshtein
import numpy as np


if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

def retrieve_action(response):
    match = re.search(r'\b(Accelerate|Hold|Decelerate)\b', response, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float | None]:
    """
    Reward function that checks if the completion is the same as the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    """
    if not is_math_verify_available():
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):

        content_sol_dist = Levenshtein.distance(content, sol)

        similarity = 1 - content_sol_dist / max(len(content), len(sol))

        sim_reward = similarity ** 6

        pred_action = retrieve_action(content)
        actual_action = retrieve_action(sol)

        acc_reward = (pred_action == actual_action) - 0.5

        reward = sim_reward + acc_reward

        rewards.append(reward)

        # print completions and reward
    # print("Completions: ", contents)
    # print(f"Avg Reward: {np.mean(rewards)}, Rewards: {rewards}")
    print(f"Avg Reward: {np.mean(rewards)}.")
    # print("===============================")

    return rewards

class DictRewardGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, action_reward_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert action_reward_dict is not None, "action_reward_dict must be provided"
        self.action_reward_dict = action_reward_dict

    def _extract_prompt_id(self, prompt):
        match = re.match(r"<id: ([^>]+)>", prompt.strip())
        return match.group(1) if match else None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, 1, device=device)
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            prompt_id = self._extract_prompt_id(prompt)
            completion_str = completion if isinstance(completion, str) else completion[0]["content"]
            action = self.get_taken_action_from_completion(completion_str)
            reward = self.action_reward_dict.get(prompt_id, {}).get(action, -3.0)
            # count the characters of the completion
            completion_length = len(completion_str)
            if completion_length > len('#Accelerate.#'):
                reward += -0.5
            rewards[i, 0] = reward
        
            # print(f"Prompt: {prompt}")
            print(f"ID: {prompt_id} Completion: {completion_str} - Action: {action} - Reward: {reward}")
            # print("=================")

        # print("Completions:", completions)
        print("Rewards:", rewards.flatten().tolist())

        return rewards

    def get_taken_action_from_completion(self, completion):
        completion_str = completion if isinstance(completion, str) else completion[0]["content"]
        m = re.search(r"\b(Accelerate|Hold|Decelerate)\b", completion_str, flags=re.IGNORECASE)
        return m.group(1).capitalize() if m else None


def setup_grpo_config(output_dir):
    return GRPOConfig(
    output_dir=output_dir,
    logging_steps=1,
    learning_rate=1e-5,               # small LR for RL on LoRA
    lr_scheduler_type="constant",
    weight_decay=0.0,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    num_generations=4,                # G: group size per prompt
    max_prompt_length=1024,
    max_completion_length=9,          # exactly one token
    remove_unused_columns=False,
    do_train=True,
    beta=0.05,
    num_train_epochs=1,
    report_to=["wandb"],
    temperature=1.0
    # use_vllm=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/data/Iman/shared_LLM_models/Qwen2.5-Math-7B/")
    parser.add_argument("--data_file", default="/data/Iman/datasets/ownship_prompts_with_action_rewards_dataNov17.jsonl")
    parser.add_argument("--output_dir", default="/data/Iman/finetuned_models/grpo-qwen-atc4/")
    parser.add_argument("--no_train", action="store_true")
    args = parser.parse_args()

    print("Starting qwen_ft_grpo setup...")
    check_environment()

    # Ensure model dir exists (download if needed is omitted here; follow notebook pattern if desired)
    tokenizer, base_model = load_model_and_tokenizer(args.model_dir)
    model = setup_lora(base_model)

    # ds_train, ds_test, prompt_action_rewards_dict = load_and_prepare_dataset(args.data_file)
    ds = load_and_prepare_dataset_(args.data_file)

    cfg = setup_grpo_config(args.output_dir)

    # trainer = DictRewardGRPOTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     args=cfg,
    #     train_dataset=ds_train,
    #     eval_dataset=ds_test,
    #     action_reward_dict=prompt_action_rewards_dict,
    #     reward_funcs=grpo_reward_function,
    # )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=accuracy_reward,
        train_dataset=ds,
        args=cfg
    )

    if not args.no_train:
        trainer.train()

        os.makedirs(args.output_dir, exist_ok=True)
        # trainer.save_model(os.path.join(args.output_dir, "merged_grpo_qwen_model"))
        model.save_pretrained(os.path.join(args.output_dir, "grpo_qwen_lora_adapter"))
        print("Saved merged model and LoRA adapter")
    else:
        print("Training skipped (--no_train)")


if __name__ == "__main__":
    main()
