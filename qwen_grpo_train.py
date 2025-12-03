#!/usr/bin/env python
# coding: utf-8
"""
GRPO Training Script for Qwen2.5-Math-7B with LoRA
Adapted from Qwen_FT_GRPO_simple.ipynb
"""

import torch
import torch.nn as nn
import os
import sys
import json
import random
import time
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm
from trl.import_utils import is_math_verify_available
import Levenshtein
import numpy as np
import wandb

# Optional DeepSpeed
try:
    import deepspeed
    import torch.distributed as dist
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available, running in single-GPU mode")

# ===============================
# Configuration
# ===============================
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Model and training configuration
model_path = "/data/Iman/shared_LLM_models/Qwen2.5-Math-7B"
dataset_dir = "/data/Iman/datasets/"
Q_batch_size = 1
beta = 0.03
num_pre_Q = 3
all_steps = 2000
max_prompt_length = 1024
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
save_dir = "/data/Iman/finetuned_models/grpo-qwen-atc5"

# DeepSpeed configuration
ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size * num_pre_Q,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-6}
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

system_prompt = """You are an airspace tactical deconfliction assistant. At each time step, an ownship agent is approaching a bottleneck waypoint, such as merging or intersections points, where other agents (intruders) are approaching as well. Based on the information of the intruders, 
the ownship should take an action to avoid collisions. The ownship agent only has access to the information of the front intruders, but there might be other intruders behind the ownship. Your task is to help the ownship aircraft avoid collisions with front intruder aircraft by 
suggesting appropriate speed adjustments. The ownship cannot unnecessarily decelerate since it might occlude the airspace for other agents behind it. Now, given the ownship state: [v, v_min, v_max delta_d1, delta_d2, delta_dwpt1, delta_dwpt2, delta_v1, delta_v2, same_route1, same_route2], 
v: ownship speed(m/s), v_min: ownship minimum speed(m/s), v_max: ownship maximum speed(m/s), d1: the absolute distance between the first front intruder (if any) and ownship(m), d2: the absolute distance between the second front intruder (if any) and ownship(m),   
delta_dwpt1: the absolute difference between the distance to the next waypoint between the first front intruder (if any) and ownship(m),  delta_dwpt2: the absolute difference between the distance to the next waypoint between the second front intruder (if any) and ownship(m), 
delta_v1: signed speed difference between the first front intruder and ownship(m/s), delta_v2: signed speed difference between the second front intruder and ownship(m/s), same_route1: whether the first front intruder is on the same route as the ownship (1: yes, 0: no), 
same_route2: whether the second front intruder is on the same route as the ownship (1: yes, 0: no). If one of the values is None, it means there is no such front intruder. Based on the above information, decide the ownship action: Decelerate / Hold / Accelerate. 
Never include reasoning, explanations, or any text other than the single chosen action.\n """

# ===============================
# Helper Functions
# ===============================

def verify_cuda():
    """Verify CUDA availability and print device information"""
    print("=" * 50)
    print("CUDA Information:")
    print(f"  Torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 50)
    
    assert torch.cuda.is_available(), "CUDA not available"


def to_grpo_row(ex):
    """Convert dataset row to GRPO format"""
    # prompt = (
    #     f"### System:\n{ex['system_prompt'].strip()}\n"
    #     f"### User:\n{ex['user_prompt'].strip()}. Always start your response with: The recommended action is: \n"
    #     f"### Assistant: "
    # )
    prompt = (
        f"### User:\n{ex['user_prompt'].strip()}. Always start your response with: The recommended action is: \n"
        f"### Assistant: "
    )
    prompt_response = ex['prompt_response']
    action = re.search(r"\b(Accelerate|Hold|Decelerate)\b", prompt_response, flags=re.IGNORECASE)
    action = action.group(1).capitalize() if action else None
    solution = "The recommended action is: " + action + "."

    return {
        "prompt": [{'content': prompt, 'role': 'user'}],
        "solution": solution,
    }

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

def sim_reward(answer: str, desired_answer: str):
    distance = Levenshtein.distance(answer, desired_answer)
    similarity = 1 - distance / max(len(answer), len(desired_answer))
    sim_reward = similarity ** 6
    return sim_reward

def action_reward(answer: str, desired_answer: str):
    pred_action = retrieve_action(answer)
    actual_action = retrieve_action(desired_answer)

    acc_reward = (pred_action == actual_action) - 0.5

    return acc_reward


def reward_correct(item, answer):
    """Calculate correctness reward"""
    from math_verify import parse, verify, ExprExtractionConfig
    
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1


def reward_format(item, answer):
    """Calculate format reward"""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


def get_per_token_logps(logits, input_ids):
    """Calculate per-token log probabilities"""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


# ===============================
# Main Training Functions
# ===============================

def gen_answers(prompts, tokenizer, gen_model, generation_config):
    """Generate answers for given prompts"""
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x}
        ], tokenize=False, add_generation_prompt=True))
    
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, 
                          padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    
    if prompt_length > max_prompt_length:
        return []
    
    # Move inputs to the correct device
    device = gen_model.device if hasattr(gen_model, 'device') else next(gen_model.parameters()).device
    tip_inputs = {k: v.to(device) for k, v in tip_inputs.items()}
    
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers


def gen_samples(inputs, tokenizer, gen_model, generation_config):
    """Generate samples with rewards"""
    prompts = [x["Q"] for x in inputs]
    desired_answers = [x["A"] for x in inputs]
    answers = gen_answers(prompts, tokenizer, gen_model, generation_config)
    
    if len(answers) == 0:
        return None, None, None, None
    
    rewards = []
    for i, inp in enumerate(inputs):
        desired_answer = desired_answers[i]
        for a in answers[i * num_pre_Q:(i + 1) * num_pre_Q]:

            # print("answer: ", a)
            # compute similarity reward
            sim_rew = sim_reward(a, desired_answer)
            # compute action reward
            act_reward = action_reward(a, desired_answer)
                                    
            rewards.append(sim_rew + act_reward)
    
    prompts_text = [tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x}
    ], tokenize=False, add_generation_prompt=True) for x in prompts]
    
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, 
                             padding_side="left", add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, 
                          padding_side="right", add_special_tokens=False)["input_ids"]
    
    return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers


def generate_mode(QAs, gen_model, tokenizer, generation_config, num=4, rank=0):
    """Generate training batches"""
    batches = []
    
    for ii in range(num):
        qas = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers = gen_samples(qas, tokenizer, gen_model, generation_config)
        
        if prompt_inputs is None:
            continue
        
        # if rank == 0:
        #     print(f'Sample {ii} rewards:', rewards)
        #     if ii == 5:
        #         print('Sample answer:', answers[0])
        
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue
        
        # Store original rewards before normalizing
        original_rewards = rewards.clone()
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        
        batch = {
            'plen': prompt_length,
            'inputs': merged_ids,
            'rewards': rewards,
            'original_rewards': original_rewards,
        }
        
        if compute_gen_logps:
            with torch.inference_mode():
                device = gen_model.device if hasattr(gen_model, 'device') else torch.device('cuda')
                mids = merged_ids.to(device)
                
                if hasattr(gen_model, 'module'):
                    out = gen_model.module(mids)
                else:
                    out = gen_model(mids)
                
                gen_logps = get_per_token_logps(out.logits[:, :-1, :], mids[:, 1:])
                batch['refs'] = gen_logps[:, prompt_length - 1:].cpu()
                batch['gen_logps'] = batch['refs'].clone()
        
        batches.append(batch)
    
    return batches


def GRPO_step(batch, model, tokenizer, device, optimizer=None):
    """Perform one GRPO training step"""
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(device)
    advantages = batch['rewards'].to(device).unsqueeze(1)
    
    logits = model(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Backward pass
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss


# ===============================
# Main Training Loop
# ===============================

def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='GRPO Training')
    parser.add_argument('--use-deepspeed', action='store_true', help='Use DeepSpeed for training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--load-in-8bit', action='store_true', help='Load model in 8-bit precision')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs with model parallelism')
    parser.add_argument('--gpus', type=str, default='0,1', help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    args = parser.parse_args()
    
    use_deepspeed = args.use_deepspeed and DEEPSPEED_AVAILABLE
    use_deepspeed = False
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Verify CUDA
    verify_cuda()
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "_attn_implementation": "sdpa",
    }
    
    if args.multi_gpu:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
        model_kwargs["device_map"] = "auto"
        max_memory = {
            gpu_id: f"{int(torch.cuda.get_device_properties(gpu_id).total_memory * 0.9 / 1e9)}GiB"
            for gpu_id in gpu_ids
        }
        model_kwargs["max_memory"] = max_memory
        print(f"Using multi-GPU with device_map='auto', max_memory: {max_memory}")
    elif args.load_in_8bit:
        print("Loading model in 8-bit mode to save memory...")
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = {"": args.gpu}
    else:
        model_kwargs["device_map"] = {"": args.gpu}
    
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        base_model.gradient_checkpointing_enable()
    
    # Add LoRA
    model = get_peft_model(base_model, LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    ))
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.4f}%)")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_dir + "ownship_prompts_with_action_rewards_dataNov17.jsonl", split='train')
    ds_train = dataset.map(to_grpo_row, remove_columns=dataset.column_names)
    
    QAs = []
    for sample in ds_train:
        QAs.append({'Q': sample['prompt'][0]['content'], 'A': sample['solution']})
    
    print(f"Dataset size: {len(QAs)}")
    
    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=9,
        do_sample=True,
        temperature=0.9,
        num_return_sequences=num_pre_Q,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Initialize training setup
    if use_deepspeed:
        print("Initializing DeepSpeed...")
        engine, optimizer, _, _ = deepspeed.initialize(
            config=ds_config,
            model=model,
            model_parameters=model.parameters()
        )
        device = torch.device(f"cuda:{engine.local_rank}")
        torch.cuda.set_device(device)
        engine.device = device
        engine.module.to(device)
        
        print(f"Engine device: {engine.device}, local_rank: {engine.local_rank}")
        print(f"Model device (sample param): {next(engine.module.parameters()).device}")
        
        gen_model = engine
        train_model = engine
        rank = torch.distributed.get_rank()
    elif args.multi_gpu:
        print("Initializing multi-GPU training with model parallelism...")
        device = torch.device("cuda")  # Primary device, model handles distribution via device_map
        print(f"Primary device: cuda:0")
        print("Model distributed across GPUs with device_map='auto'")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        
        # Check memory on each GPU
        gpu_list = [int(g) for g in args.gpus.split(',')]
        for gpu in gpu_list:
            allocated = torch.cuda.memory_allocated(gpu) / 1e9
            reserved = torch.cuda.memory_reserved(gpu) / 1e9
            print(f"GPU {gpu} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        gen_model = model
        train_model = model
        rank = 0
    else:
        print("Initializing single-GPU training...")
        device = torch.device(f"cuda:{args.gpu}")
        
        # Only move to device if not using 8-bit (8-bit already handles device placement)
        if not args.load_in_8bit:
            model.to(device)
        
        model.device = device
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        
        print(f"Model device: {device}")
        print(f"Model device (sample param): {next(model.parameters()).device}")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(args.gpu) / 1e9
            reserved = torch.cuda.memory_reserved(args.gpu) / 1e9
            print(f"GPU Memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
        
        gen_model = model
        train_model = model
        rank = 0
    
    # Initialize wandb
    if rank == 0:
        wandb.init(project="GRPO_Training", name="Qwen_GRPO_Run", config={
            "model": model_path,
            "batch_size": Q_batch_size,
            "num_pre_Q": num_pre_Q,
            "learning_rate": 1e-6,
            "all_steps": all_steps,
        })
    
    # Training loop
    print("Starting training...")
    progress = range(1, all_steps + 1)
    if rank == 0:
        progress = tqdm(progress)
    
    for step in progress:
        # Generate batches
        batches = generate_mode(QAs, gen_model, tokenizer, generation_config, num=10, rank=rank)
        
        if not batches:
            print(f"Warning: No valid batches generated at step {step}")
            continue
        
        # Train on each batch
        total_loss = 0.0
        all_original_rewards = []
        for batch in batches:
            if use_deepspeed:
                loss = GRPO_step(batch, engine, tokenizer, device)
                engine.backward(loss)
                engine.step()
            else:
                loss = GRPO_step(batch, train_model, tokenizer, device, optimizer)
            
            total_loss += loss.item()
            
            # Collect original rewards for step average
            all_original_rewards.append(batch['original_rewards'])
            
            # # Print average reward for this batch
            # if rank == 0:
            #     avg_reward = batch['original_rewards'].mean().item()
            #     print(f"Average reward for batch: {avg_reward:.4f}")
        
        avg_loss = total_loss / len(batches) if batches else 0.0
        
        # Calculate average reward per step
        if all_original_rewards:
            step_avg_reward = torch.cat(all_original_rewards).mean().item()
            if rank == 0:
                print(f"Average reward for step {step}: {step_avg_reward:.4f}")
                wandb.log({"loss": avg_loss, "reward": step_avg_reward}, step=step)
        
        if rank == 0:
            if isinstance(progress, tqdm):
                progress.set_description(f"Loss: {avg_loss:.6f}")
            else:
                print(f"Step {step}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if step % save_steps == 0:
            if use_deepspeed:
                dist.barrier()
            
            # if rank == 0:
            print(f'\nSaving model at step {step}...')
            save_name = f"{save_dir}/step_{step}"
            
            if use_deepspeed:
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
            else:
                model.save_pretrained(save_name)
            
            tokenizer.save_pretrained(save_name)
            print(f'Model saved to {save_name}')
            
            if use_deepspeed:
                dist.barrier()
    
    print("Training complete!")
    
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
