import os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

EVALUATION_REPORT_FOLDER = "./evaluation_reports"
os.makedirs(EVALUATION_REPORT_FOLDER, exist_ok=True)


#### 1. Load the dataset
data_file = "test_grpo_model_data.jsonl" #"train_data_new_final_prompt_with_number2.jsonl" # "/data/Iman/datasets/ownship_prompts_data.jsonl" #  # "filteredBalanced_ownship_prompts_data_Nov10.jsonl"
dataset = load_dataset("json", data_files=data_file)
test_dataset = dataset["train"] #.train_test_split(test_size=0.001, seed=7, shuffle=True, keep_in_memory=False)
# test_dataset = dataset["test"]

#### 2. Load the base model + tokenizer + LoRA weights for inference
BASE_MODEL_DIR = "/data/Iman/shared_LLM_models/Qwen2.5-Math-7B"
LORA_WEIGHTS_DIR = "/data/Iman/finetuned_models/grpo2/grpo_model"
# LORA_WEIGHTS_DIR = "/data/Iman/finetuned_models/lora-qwen-atc-finetuned-lora-adapters"

# get the model name from the LoRA_WEIGHTS_DIR
MODEL_NAME =  os.path.basename(LORA_WEIGHTS_DIR)
REPORT_SUBFOLDER = os.path.join(EVALUATION_REPORT_FOLDER, MODEL_NAME)
os.makedirs(REPORT_SUBFOLDER, exist_ok=True)



tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=False)
tokenizer.padding_side = "left"  # Set left padding for decoder-only models

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR, 
    trust_remote_code=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# attach LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS_DIR,
    # device_map="auto",
    # torch_dtype=torch.float16,
)
# model = base_model
model.eval()

#### 3. Tokenize the dataset
def format_prompt(system_prompt, user_prompt, prompt_response):
    action = re.search(r'\b(Accelerate|Decelerate|Hold)\b', prompt_response, re.IGNORECASE)
    gt_action = action.group(1) if action else "Unknown"
    # user_prompt += "\nYour response should start with 'The predicted action is: ' followed by one of the actions: Decelerate, Hold, or Accelerate.\n"
    prompt = (f"### System:\n{system_prompt.strip()}\n\n"
            f"### User:\n{user_prompt.strip()}\n\n"
            f"### Assistant:\n{prompt_response.strip()}\n"
    )
    return prompt, gt_action


# tokenize for inference 
def tokenize_for_eval(batch, max_length=1024):
    prompts = []
    gt_actions = []
    system_prompts = batch['system_prompt']
    user_prompts = batch['user_prompt']
    prompt_responses = batch['prompt_response']
    for sys, user, prompt_response in zip(system_prompts, user_prompts, prompt_responses):
        # example = {
        #     'system_prompt': sys,
        #     'user_prompt': user,
        #     'prompt_response': prompt_response
        # }
        prompt, gt_action = format_prompt(sys, user, prompt_response)
        prompts.append(prompt)
        gt_actions.append(gt_action)

    # print("len prompts:", len(prompts))
    tokenized_batch = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    if "attention_mask" not in tokenized_batch:
        tokenized_batch["attention_mask"] = (tokenized_batch["input_ids"] != tokenizer.pad_token_id).long()
    
    tokenized_batch['gt_actions'] = gt_actions
    tokenized_batch['prompts'] = prompts
    return tokenized_batch

# batchwise evaluation loop
BATCH = 32
results = []
num_batches = (len(test_dataset) + BATCH - 1) // BATCH

for batch_id in range(num_batches):
    start_idx = batch_id * BATCH
    end_idx = min((batch_id + 1) * BATCH, len(test_dataset))

    batch = test_dataset[start_idx:end_idx]
    # print(batch)
    tokenized_batch = tokenize_for_eval(batch)
    input_ids = tokenized_batch['input_ids'].to(model.device)
    attention_mask = tokenized_batch['attention_mask'].to(model.device)

    # print(input_ids.shape, attention_mask.shape)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            # temperature=0.0,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # num_beams=1,
        )

    completions = []
    # for j in range(len(batch)):
    #     generated_text = tokenizer.decode(gen_ids[j], skip_special_tokens=True)
    #     print("Full generated text:", generated_text)
    #     completions.append(generated_text)
    #     results.append({
    #         # 'prompt': tokenized_batch['prompts'][j],
    #         'ground_truth_action': tokenized_batch['gt_actions'][j],
    #         'generated_text': generated_text
    #     })
    decoded = tokenizer.batch_decode(
        gen_ids[:, input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    # print("Decoded completions:", decoded)
    
    
    for text, prompt in zip(decoded, tokenized_batch['prompts']):
        # print("Full generated text:", text)
        completions.append(text)

    # store results
    for gt, pred in zip(tokenized_batch['gt_actions'], completions):
        results.append({
            'ground_truth_action': gt,
            'predicted_action': pred
        })

    print(f"Processed {min(len(results), len(test_dataset))}/{len(test_dataset)} examples.")

# compute accuracy
def extract_pred_action(text):
    match = re.search(r'\b(Accelerate|Decelerate|Hold)\b', text, re.IGNORECASE)
    return match.group(1) if match else "Unknown"

ACTIONS = ["Accelerate", "Decelerate", "Hold", "Unknown"]
y_true = []
y_pred = []

correct = 0
total = len(results)
for res in results:
    gt = res['ground_truth_action']
    pred_raw = res['predicted_action']

    gt_action = gt.capitalize() if gt in ACTIONS else gt
    pred_action = extract_pred_action(pred_raw).capitalize()

    y_true.append(gt_action)
    y_pred.append(pred_action)

    if pred_action.lower() == gt.lower():
        correct += 1


acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=ACTIONS)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=ACTIONS, zero_division=0)

# save detailed classification report
report = classification_report(y_true, y_pred, labels=ACTIONS, zero_division=0)
with open(REPORT_SUBFOLDER + "/classification_report.txt", "w") as f:
    f.write(report)


print(f"Accuracy: {correct}/{total} ({correct/total*100:.2f}%) - {acc*100:.2f}%")

# save results to a file
import json
with open(REPORT_SUBFOLDER + "/evaluation_results.jsonl", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation results saved to evaluation_results.jsonl")