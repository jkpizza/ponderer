import transformers
import os
import numpy as np
from trl import SFTTrainer
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction
)
import re
import torch.nn as nn
import random
import wandb

class LatentTokenWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, latent_token_id: int, start_latent_id: int, end_latent_id: int):
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.ignore_token_ids = {latent_token_id, start_latent_id, end_latent_id}

        # Expose HF/Accelerate hints so sharding is preserved
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.hf_device_map = getattr(base_model, "hf_device_map", None)
        self.is_parallelizable = getattr(base_model, "is_parallelizable", False)
        self.model_parallel = getattr(base_model, "model_parallel", False)

    # Prevent Trainer from collapsing shards; allow dtype cast only
    def to(self, device=None, dtype=None, non_blocking=False, **kwargs):
        if self.hf_device_map:
            if dtype is not None:
                for p in self.base_model.parameters():
                    p.data = p.data.to(dtype=dtype)
            return self
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking, **kwargs)

    # Minimal proxies Trainer expects
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def gradient_checkpointing_disable(self):
        return self.base_model.gradient_checkpointing_disable()
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
    def set_input_embeddings(self, new_emb):
        return self.base_model.set_input_embeddings(new_emb)
    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)
    def push_to_hub(self, *args, **kwargs):
        return self.base_model.push_to_hub(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Fast path: if no latents, defer entirely
        if input_ids is None or not (input_ids == self.latent_token_id).any():
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        if labels is not None:
            labels = labels.clone()
            for tid in self.ignore_token_ids:
                labels[input_ids == tid] = -100

        device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        embed = self.base_model.get_input_embeddings()
        filled = embed(input_ids).clone()
        B, T = input_ids.shape

        latent_mask = (input_ids == self.latent_token_id)
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        last_lat_per_row = torch.where(latent_mask, idxs, torch.full_like(idxs, -1)).max(dim=1).values
        max_last_lat = int(last_lat_per_row.max().item())
        prev_hidden = None

        emb_list = []
        for i in range(max_last_lat + 1):
            base_emb_i = embed(input_ids[:, i])  # [B, H]
            use_latent = (input_ids[:, i] == self.latent_token_id)

            if i == 0 and use_latent.any():
                bos_id = getattr(self.base_model.config, "bos_token_id", None)
                if bos_id is not None:
                    bos_emb = embed(torch.full_like(input_ids[:, i], bos_id))
                    emb_i = torch.where(use_latent.unsqueeze(-1), bos_emb, base_emb_i)
                else:
                    emb_i = base_emb_i
            elif i > 0 and use_latent.any():
                emb_i = torch.where(use_latent.unsqueeze(-1), prev_hidden, base_emb_i)
            else:
                emb_i = base_emb_i

            emb_list.append(emb_i.unsqueeze(1))  # grows the differentiable prefix

            # Forward over current prefix to get the last hidden (no cache; keep graph)
            out = self.base_model(
                inputs_embeds=torch.cat(emb_list, dim=1),
                attention_mask=attention_mask[:, : i + 1],
                labels=labels[:, : i + 1],
                use_cache=False,
                output_hidden_states=False,
            )
            prev_hidden = out.last_hidden_state[:, -1, :]  # requires_grad=True

        # Full forward with grads on the filled embeds
        if max_last_lat + 1 < T:
            rest = embed(input_ids[:, max_last_lat + 1:])  # [B, T - (max_last_lat+1), H]
            filled = torch.cat([torch.cat(emb_list, dim=1), rest], dim=1)
        else:
            filled = torch.cat(emb_list, dim=1)
            
        return self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
    

def main():
    latent = False
    run_name = "llama3b_gsm8k_cot"

    # base_model = "meta-llama/Llama-3.1-8B-Instruct" # Llama 8B Model
    base_model = "meta-llama/Llama-3.2-3B-Instruct" # Llama 3B Model
    # base_model = "meta-llama/Llama-3.2-1B-Instruct" # Llama 1B Model

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        # device_map="auto",
        device_map="balanced_low_0",
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if latent:
        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")
        start_token_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_token_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        model.resize_token_embeddings(len(tokenizer))
    
        # Wrap base model with latent token wrapper
        model = LatentTokenWrapper(model, latent_id, start_token_id, end_token_id)
    
    def formatting_prompts_func(example, latent=False, max_latents=10):
        
        question = example["question"]
        answer = example["answer"]
        if latent:
            num_latents = random.randint(0, max_latents) # random number of latents between 0 and max_latents
            text = f"Question: {question}\n<|start-latent|>{'<|latent|>'*num_latents}<|end-latent|>### Answer: {answer}{tokenizer.eos_token}"        
        else:
            text = f"Question: {question}\n### Answer: {answer}{tokenizer.eos_token}"        
        return text
    
    def extract_final_answer(text: str) -> str:
        """Extract the final answer from GSM8K-formatted text"""
        match = re.search(r'####\s*(\d+)', text)
        if match:
            answer = match.group(1)
            answer = answer.replace(',', '')
            return answer
        else:
            return ''

    class GSM8KMetrics:
        def __init__(self, tokenizer):  # Pass tokenizer for decoding
            self.tokenizer = tokenizer
            self.total_matches = 0
            self.total_examples = 0

        def __call__(self, eval_pred: EvalPrediction, compute_result: bool) -> dict:
            if not compute_result:
                # Accumulate per batch
                logits, label_ids = eval_pred.predictions, eval_pred.label_ids
                
                # Greedy decode predictions
                pred_ids = np.argmax(logits.cpu(), axis=-1)  # Shape: [batch_size, seq_len]
                
                # Decode to strings
                pred_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]
                label_texts = [self.tokenizer.decode(ids[ids != -100], skip_special_tokens=True) for ids in label_ids]
                
                # Extract and count matches (accumulate counters)
                pred_answers = [extract_final_answer(text) for text in pred_texts]
                label_answers = [extract_final_answer(text) for text in label_texts]
                batch_matches = sum(p == l for p, l in zip(pred_answers, label_answers))
                
                self.total_matches += batch_matches
                self.total_examples += len(pred_answers)
                return {}  # No return needed during accumulation
            
            else:
                # Final computation
                if self.total_examples == 0:
                    accuracy = 0.0
                else:
                    accuracy = self.total_matches / self.total_examples
                
                # Reset for next eval
                self.total_matches = 0
                self.total_examples = 0
                
                return {"exact_match_accuracy": accuracy}
       
    ds = load_dataset("gsm8k", "main") # GSM8K Dataset
    # ds = load_dataset("nlile/hendrycks-MATH-benchmark") # MATH Dataset
    train_ds = ds["train"]
    eval_ds = ds["test"]
    # eval_ds = eval_ds.select(range(100))
    
    training_args = TrainingArguments(
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        group_by_length=True,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        eval_strategy="steps",
        # eval_accumulation_steps=10,
        batch_eval_metrics=True,
        eval_on_start=True,
        eval_steps=0.10,
        logging_steps=1,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results/",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True}, #must be false for DDP
    )
    
    
    wandb.init(project="ponderer", name=run_name)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_prompts_func,
        compute_metrics=GSM8KMetrics(tokenizer),
    )
    
    trainer.train()
    
if __name__ == "__main__":
    main()