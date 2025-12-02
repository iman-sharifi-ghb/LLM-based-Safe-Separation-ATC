#!/usr/bin/env python3
"""
GRPO Fine-tuning Script for Qwen ATC Model
This script fine-tunes a Qwen model using Group Relative Policy Optimization (GRPO)
for Air Traffic Control decision making tasks.
"""

import torch
import re
import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import argparse


def gpu_memory():
    """Display GPU memory usage information."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        encoding="utf-8"
    )
    lines = result.stdout.strip().split("\n")
    for idx, line in enumerate(lines):
        used, total = line.split(", ")
        print(f"GPU {idx}: {used} MB / {total} MB used")


def check_environment():
    """Check and display system information."""
    print("Torch version:", torch.__version__)
    print("Torch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("Current directory:", os.getcwd())
    print("\n🔎 GPU Memory before model load:")
    gpu_memory()


def load_model_and_tokenizer(model_path):
    """Load tokenizer and base model."""
    print(f"🚀 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="left", 
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print("✅ Model ready for inference!")
    return tokenizer, base_model


def setup_lora_model(base_model):
    """Setup LoRA configuration and wrap the base model."""
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=32,  # alpha (scaling factor)
        lora_dropout=0.05,  # dropout
        target_modules=[
            "up_proj", "down_proj", "gate_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    
    # Print trainable parameter names
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  - {name}")
    
    return model


def load_and_prepare_dataset(data_file):
    """Load and prepare dataset for GRPO training."""
    print(f"Loading dataset from: {data_file}")
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_file)
    ds = dataset["train"].train_test_split(test_size=0.2, seed=42, shuffle=True, keep_in_memory=False)
    print("Dataset split:", ds)
    
    def to_grpo_row(ex):
        """Convert dataset row to GRPO format."""
        prompt = (
            f"### System:\n{ex['system_prompt'].strip()}\n\n"
            f"### User:\n{ex['user_prompt'].strip()}\n\n"
            f"### Assistant:\n The recommended action for ownship is to "
        )
        return {
            "prompt": prompt,
            "action_rewards": ex["action_rewards"],
        }
    
    def add_prompt_id(example, idx):
        """Add prompt ID at the beginning of the prompt."""
        prompt_id = f"<id: p{idx}>"
        example["prompt"] = prompt_id + " " + example["prompt"]
        return example
    
    # Process datasets
    ds_train = ds["train"].map(to_grpo_row, remove_columns=ds["train"].column_names)
    ds_test = ds["test"].map(to_grpo_row, remove_columns=ds["test"].column_names)
    
    # Add prompt IDs
    ds_train = ds_train.map(add_prompt_id, with_indices=True)
    ds_test = ds_test.map(add_prompt_id, with_indices=True)
    
    # Create dictionary mapping prompt IDs to action rewards
    prompt_action_rewards_dict = {}
    for row in ds_train:
        match = re.match(r"<id: ([^>]+)>", row['prompt'].strip())
        if match:
            prompt_id = match.group(1)
            prompt_action_rewards_dict[prompt_id] = row['action_rewards']
    
    print(f"Created dictionary with {len(prompt_action_rewards_dict)} entries")
    print("Sample entries:")
    for i, (key, value) in enumerate(list(prompt_action_rewards_dict.items())[:3]):
        print(f"  {key}: {value}")
    
    return ds_train, ds_test, prompt_action_rewards_dict


def extract_action(text):
    """
    Search for (Decelerate/Hold/Accelerate) actions in the text and output it.
    
    Args:
        text (str): Text like "The recommended action for the ownship is Accelerate"
        
    Returns:
        str: Action like "Accelerate" or None if not found
    """
    action = re.search(r"\b(Accelerate|Hold|Decelerate)\b", text, flags=re.IGNORECASE)
    final_action = action.group(1).capitalize() if action else None
    return final_action


def grpo_reward_function(completions, action_rewards=None, **kwargs):
    """
    Calculate reward for given completions.
    
    Args:
        completions: List of completion strings generated by the model
        action_rewards: List of action reward dictionaries
        
    Returns:
        List of rewards corresponding to each completion
    """
    if action_rewards is None:
        action_rewards = [{"Accelerate": 0, "Hold": 1, "Decelerate": 0}] * len(completions)
    
    rewards = []
    for comp, action_reward in zip(completions, action_rewards):
        action = extract_action(comp)
        if action is None:
            rewards.append(-3)
        else:
            rewards.append(action_reward.get(action, 0.0))
    return rewards


class DictRewardGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer that uses action reward dictionaries."""
    
    def __init__(self, *args, action_reward_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert action_reward_dict is not None, "action_reward_dict must be provided"
        self.action_reward_dict = action_reward_dict

    def _extract_prompt_id(self, prompt):
        """Extract prompt ID from prompt string."""
        match = re.match(r"<id: ([^>]+)>", prompt.strip())
        if match:
            return match.group(1)
        else:
            return None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Calculate rewards for batch of prompts and completions."""
        device = self.accelerator.device
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, 1, device=device)
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            prompt_id = self._extract_prompt_id(prompt)
            completion_str = completion if isinstance(completion, str) else completion[0]["content"]
            action = self.get_taken_action_from_completion(completion_str)
            reward = self.action_reward_dict.get(prompt_id, {}).get(action, -3.0)
            rewards[i, 0] = reward
        
        print("Rewards:", rewards.flatten().tolist())
        return rewards
    
    def get_taken_action_from_completion(self, completion):
        """Extract action from completion text."""
        completion_str = completion if isinstance(completion, str) else completion[0]["content"]
        action = re.search(r"\b(Accelerate|Hold|Decelerate)\b", completion_str, flags=re.IGNORECASE)
        final_action = action.group(1).capitalize() if action else "Hold"
        return final_action


def setup_grpo_config(output_dir):
    """Setup GRPO training configuration."""
    return GRPOConfig(
        output_dir=output_dir,
        logging_steps=5,
        learning_rate=2e-4,               # small LR for RL on LoRA
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,                # G: group size per prompt
        max_prompt_length=1024,
        max_completion_length=3,
        beta=0.02,                        # enable a small KL to the ref policy
        scale_rewards="batch",            # stabler than per-group
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        # Generation constraints
        top_k=None,
        min_p=None,
        top_p=None,
        temperature=1.0,
        report_to=["wandb"],
        do_train=True,
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GRPO Fine-tuning for Qwen ATC Model")
    parser.add_argument("--model_path", type=str, 
                       default="/data/Iman/finetuned_models/merged-qwen-atc-model",
                       help="Path to the base model")
    parser.add_argument("--data_file", type=str,
                       default="/data/Iman/datasets/ownship_prompts_with_action_rewards_dataNov17.jsonl",
                       help="Path to the training data file")
    parser.add_argument("--output_dir", type=str,
                       default="/data/Iman/finetuned_models/grpo-qwen-atc/",
                       help="Output directory for trained model")
    parser.add_argument("--no_train", action="store_true",
                       help="Skip training (for testing setup)")
    
    args = parser.parse_args()
    
    # Check environment
    print("="*50)
    print("GRPO Fine-tuning for Qwen ATC Model")
    print("="*50)
    check_environment()
    
    # Load model and tokenizer
    tokenizer, base_model = load_model_and_tokenizer(args.model_path)
    
    # Setup LoRA
    model = setup_lora_model(base_model)
    
    # Load and prepare dataset
    ds_train, ds_test, prompt_action_rewards_dict = load_and_prepare_dataset(args.data_file)
    
    # Setup GRPO configuration
    cfg = setup_grpo_config(args.output_dir)
    
    # Create trainer
    print("\n🚀 Setting up GRPO trainer...")
    trainer = DictRewardGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=ds_train,
        action_reward_dict=prompt_action_rewards_dict,
        reward_funcs=grpo_reward_function,
    )
    
    if not args.no_train:
        # Start training
        print("\n🎯 Starting GRPO training...")
        trainer.train()
        
        # Save models
        print("\n💾 Saving models...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save the merged model
        merged_model_path = os.path.join(args.output_dir, "merged_grpo_qwen_model")
        trainer.save_model(merged_model_path)
        print(f"✅ Merged model saved to: {merged_model_path}")
        
        # Save the LoRA adapters only
        lora_adapter_path = os.path.join(args.output_dir, "grpo_qwen_lora_adapter")
        model.save_pretrained(lora_adapter_path)
        print(f"✅ LoRA adapters saved to: {lora_adapter_path}")
        
        print("\n🎉 Training completed successfully!")
    else:
        print("\n⚠️ Training skipped (--no_train flag used)")
        print("Setup completed successfully!")


if __name__ == "__main__":
    main()