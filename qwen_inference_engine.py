import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import Markdown, display
from torch.nn.attention import SDPBackend, sdpa_kernel

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.eval()
            print("✅ Model and tokenizer loaded successfully")
        else:
            raise ValueError("Either provide model_path or both tokenizer and model instances")
        
        # Set generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": False,
            "temperature": 0.0,
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
        if match:
            action = match.group(1).capitalize()
        else:   
            action = "Hold"
            print("⚠️ No valid action found in output; defaulting to 'Hold'")
        return action
    
    def format_prompt(self, system_prompt, user_prompt):
        return (f"### System:\n{system_prompt.strip()}\n\n"
                f"### User:\n{user_prompt.strip()} Always start your response with: {"The recommended action is: "}\n"
                f"### Assistant:\n"
        )
    
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

        
        # Create chat-formatted prompt
        formatted_prompt = self.format_prompt(system_prompt, user_prompt)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != self.tokenizer.pad_token_id).long()
        
        
        # Generate response
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION): # torch.no_grad():
            output_ids = self.model.generate(**inputs, **self.generation_kwargs)
        

        raw_output = self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        assistant_response = raw_output.split("### Assistant:")[-1].strip()
        print("||| Assistant Response:", assistant_response)
        # Extract action
        action = self.extract_action(assistant_response)
        
        if verbose:
            display(Markdown(f"**Raw Output:** {assistant_response}"))
            print(f"**Extracted Action:** {action}")
        
        return {
            'action': action,
            'raw_output': assistant_response
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
            # scenarios = [
            #     prompt + " Your output must contain exactly one word and no punctuation." 
            #     for prompt in batch_prompts
            # ]
            batched_prompt_formatted = [
                self.format_prompt(system_prompt, user_p) for user_p in batch_prompts
            ]
            current_batch_size = len(batched_prompt_formatted)
            # print("Batch size:" , current_batch_size)
            # Create chat-formatted prompts for Qwen
            # messages_batch = [
            #     [
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": scenario}
            #     ]
            #     for scenario in batch_prompts
            # ]
            
            # rendered_prompts = [
            #     self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            #     for m in messages_batch
            # ]
            
            # Tokenize as one padded batch
            enc = self.tokenizer(
                batched_prompt_formatted,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # check whether attention mask is used. Note that this is batched input
            if "attention_mask" not in enc:
                enc["attention_mask"] = (enc["input_ids"] != self.tokenizer.pad_token_id).long()

            # if isinstance(enc, torch.Tensor):
            #     enc = {"input_ids": enc}
            
            # if "attention_mask" not in enc:
            #     enc["attention_mask"] = (enc["input_ids"] != self.tokenizer.pad_token_id).long()
            # .to(self.model.device)
            
            # Generate in batch
            # self.generation_kwargs["eos_token_id"] = None
            batch_generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": False,
            "temperature": 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
            with torch.inference_mode(): #, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = self.model.generate(
                    **enc, 
                    **batch_generation_kwargs,
                )

            # # decode all at once
            decoded_outputs = self.tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # post-process each decoded sample
            for i, text_output in enumerate(decoded_outputs):
                
                assistant_response = text_output.split("### Assistant:")[-1].strip()
                # print("Raw OUT:", text_output, "||| Assistant Response:", assistant_response)
                action = self.extract_action(assistant_response)
                all_actions.append(action)
                all_raw_outputs.append(assistant_response)

                if verbose:
                    display(Markdown(f"**Prompt {start_idx + i + 1}:** {assistant_response}"))
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