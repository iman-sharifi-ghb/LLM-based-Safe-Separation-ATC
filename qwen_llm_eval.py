import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import Markdown, display

class QwenInferenceEngine:
    """
    A class-based implementation for Qwen model inference with action extraction.
    Initializes model and dependencies once, then provides methods for inference.
    """
    
    def __init__(self, model_path=None, tokenizer=None, model=None):
        """
        Initialize the inference engine with model and tokenizer.
        
        Args:
            model_path (str, optional): Path to the model directory
            tokenizer (optional): Pre-loaded tokenizer instance
            model (optional): Pre-loaded model instance
        """
        
        # Initialize model and tokenizer
        if tokenizer is not None and model is not None:
            # Use provided instances
            self.tokenizer = tokenizer
            self.model = model
            print("✅ Using provided tokenizer and model instances")
        elif model_path is not None:
            # Load from path
            print(f"🚀 Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()
            print("✅ Model and tokenizer loaded successfully")
        else:
            raise ValueError("Either provide model_path or both tokenizer and model instances")
        
        # Set generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 50,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        print("🎯 QwenInferenceEngine initialized and ready for inference!")
    
    def extract_action(self, text_output):
        """
        Extract action from model output using regex.
        
        Args:
            text_output (str): Raw model output text
            
        Returns:
            str: Extracted action (Accelerate, Decelerate, or Hold)
        """
        match = re.search(r"\b(Accelerate|Decelerate|Hold)\b", text_output, flags=re.IGNORECASE)
        return match.group(1).capitalize() if match else "Hold"
    
    def single_inference(self, system_prompt, user_prompt, verbose=False):
        """
        Process a single prompt and return the extracted action.
        
        Args:
            system_prompt (str): The system prompt
            user_prompt (str): The user prompt
            verbose (bool): Whether to display detailed output
            
        Returns:
            dict: Contains 'action' and 'raw_output'
        """
        # Add instruction for single-word output
        enhanced_user_prompt = user_prompt
        
        # Create chat-formatted prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]
        
        # Apply chat template and tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(inputs, **self.generation_kwargs)
        
        # Decode only the generated part
        gen_ids = output_ids[0, inputs.shape[-1]:]
        raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Extract action
        action = self.extract_action(raw_output)
        
        if verbose:
            display(Markdown(f"**Raw Output:** {raw_output}"))
            print(f"**Extracted Action:** {action}")
        
        return {
            'action': action,
            'raw_output': raw_output
        }
    
    def batch_inference(self, system_prompt, user_prompts, batch_size=20, verbose=False):
        """
        Process multiple prompts in batches and return extracted actions.
        
        Args:
            system_prompt (str): The system prompt for all conversations
            user_prompts (list): List of user prompts to process
            batch_size (int): Number of prompts to process in each batch
            verbose (bool): Whether to display detailed outputs
            
        Returns:
            dict: Contains 'actions' and 'raw_outputs' lists
        """
        all_actions = []
        all_raw_outputs = []
        
        # Process in batches
        num_batches = (len(user_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(user_prompts))
            
            batch_prompts = user_prompts[start_idx:end_idx]
            
            if verbose:
                print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)")
            
            # Add instruction for single-word output to each prompt
            scenarios = [
                prompt
                for prompt in batch_prompts
            ]
            
            # Create chat-formatted prompts for Qwen
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scenario}
                ]
                for scenario in scenarios
            ]
            
            rendered_prompts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]
            
            # Tokenize as one padded batch
            enc = self.tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            # Generate in batch
            with torch.inference_mode():
                out = self.model.generate(**enc, **self.generation_kwargs)
            
            # Decode per row and extract final action
            input_lens = (enc["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
            
            for i in range(len(rendered_prompts)):
                gen_ids = out[i, input_lens[i]:]
                text_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                all_raw_outputs.append(text_output)
                
                if verbose:
                    display(Markdown(f"**Prompt {start_idx + i + 1}:** {text_output}"))
                
                # Extract action using regex
                action = self.extract_action(text_output)
                all_actions.append(action)
                
                if verbose:
                    print(f"[Prompt {start_idx + i + 1}] → {action}")
        
        return {
            'actions': all_actions,
            'raw_outputs': all_raw_outputs
        }
    
    def evaluate_accuracy(self, system_prompt, user_prompts, desired_responses, batch_size=20, verbose=False):
        """
        Process prompts and calculate accuracy against desired responses.
        
        Args:
            system_prompt (str): The system prompt
            user_prompts (list): List of user prompts
            desired_responses (list): List of expected responses
            batch_size (int): Batch size for processing
            verbose (bool): Whether to display detailed outputs
            
        Returns:
            dict: Contains 'actions', 'raw_outputs', 'accuracy', and 'accuracy_per_batch'
        """
        if len(user_prompts) != len(desired_responses):
            raise ValueError("user_prompts and desired_responses must have the same length")
        
        # Extract desired actions from responses
        desired_actions = []
        for response in desired_responses:
            action = self.extract_action(response)
            desired_actions.append(action)
        
        # Get predictions
        results = self.batch_inference(system_prompt, user_prompts, batch_size, verbose)
        predicted_actions = results['actions']
        
        # Calculate overall accuracy
        correct = sum(1 for pred, true in zip(predicted_actions, desired_actions) if pred == true)
        overall_accuracy = correct / len(predicted_actions)
        
        # Calculate per-batch accuracy
        accuracy_per_batch = []
        num_batches = (len(user_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(user_prompts))
            
            batch_predicted = predicted_actions[start_idx:end_idx]
            batch_desired = desired_actions[start_idx:end_idx]
            
            batch_correct = sum(1 for pred, true in zip(batch_predicted, batch_desired) if pred == true)
            batch_accuracy = batch_correct / len(batch_predicted)
            accuracy_per_batch.append(batch_accuracy)
            
            if verbose:
                print(f"Batch {batch_idx + 1} Accuracy: {batch_accuracy * 100:.2f}%")
        
        if verbose:
            print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
        
        return {
            'actions': predicted_actions,
            'raw_outputs': results['raw_outputs'],
            'accuracy': overall_accuracy,
            'accuracy_per_batch': accuracy_per_batch,
            'desired_actions': desired_actions
        }
    
    def __repr__(self):
        return f"QwenInferenceEngine(model_device={self.model.device}, tokenizer_vocab_size={len(self.tokenizer)})"

if __name__ == "__main__":
    os.environ["FLASH_ATTENTION"] = "1"

    # Example usage:
    engine = QwenInferenceEngine(model_path="/data/Iman/shared_LLM_models/Qwen2.5-Math-7B")

    result = engine.single_inference(system_prompt="Based on your knowledge, what is x?", user_prompt="x^2 + 2x + 1 = 0", 
                                    verbose=True)

    print(result)