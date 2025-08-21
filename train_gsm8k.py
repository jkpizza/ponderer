import transformers
import os
import numpy as np
from trl import SFTTrainer
from trl.trainer import SFTConfig
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
from accelerate import Accelerator
from model.latent_wrapper import LatentTokenWrapper


def main():
    # Arguments
    # ------------------------------------------------------------
    LATENT = True
    run_name = "llama1b_gsm8k_lat"

    # base_model = "meta-llama/Llama-3.1-8B-Instruct" # Llama 8B Model
    # base_model = "meta-llama/Llama-3.2-3B-Instruct" # Llama 3B Model
    base_model = "meta-llama/Llama-3.2-1B-Instruct" # Llama 1B Model
    # ------------------------------------------------------------
    
    accelerator = Accelerator()
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        # device_map="auto",
        # device_map="balanced",
        # device_map=0,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if LATENT:
        # tokenizer.add_tokens("<|start-latent|>")
        # tokenizer.add_tokens("<|end-latent|>")
        # tokenizer.add_tokens("<|latent|>")
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
        })
        start_token_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_token_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        model.resize_token_embeddings(len(tokenizer))
    
        # Wrap base model with latent token wrapper
        model = LatentTokenWrapper(model, latent_id, start_token_id, end_token_id)
    
    def formatting_prompts_func(example, latent=None, max_latents=10):
        if latent is None:
            latent = LATENT
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
        match = re.search(r'####\s*(-?[\d,]+)', text)
        if match:
            # Correct for commas, e.g., "1,600" -> "1600"
            answer = match.group(1)
            answer = answer.replace(',', '')
            return answer
        else:
            return ''
       
    ds = load_dataset("gsm8k", "main") # GSM8K Dataset
    # ds = load_dataset("nlile/hendrycks-MATH-benchmark") # MATH Dataset
    train_ds = ds["train"]
    eval_ds = ds["test"]
    # eval_ds = eval_ds.select(range(100))
    
    training_args = SFTConfig(
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        dataloader_drop_last=True,
        group_by_length=True,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        eval_strategy="no",
        # eval_accumulation_steps=10,
        # batch_eval_metrics=True,
        # eval_on_start=True,
        # eval_steps=0.10,
        max_length=1024,
        logging_steps=1,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results/",
        save_safetensors=False, # lm head.weight = embed_tokens.weight
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": False}, #must be false for DDP
        
        # TRL-specific
        dataset_num_proc=os.cpu_count(),   # parallel dataset map()
    )
    
    

    if accelerator.is_main_process:
        wandb.init(project="ponderer", name=run_name)

    # Prevent other processes from printing/logging
    accelerator.wait_for_everyone()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer, # instead of tokenizer=tokenizer
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
    
    trained = trainer.model
    to_save = trained.base_model if isinstance(trained, LatentTokenWrapper) else trained
    to_save.save_pretrained(f"./results/{run_name}", safe_serialization=False)
    tokenizer.save_pretrained(f"./results/{run_name}")
    
    
    from eval_gsm8k_v5 import batched_generate_eval
    batched_generate_eval(model, tokenizer, eval_ds, batch_size=8, max_new_tokens=1024)
    
    # # Evaluation
    # final_metrics = trainer.evaluate()
    # print('Final Evaluation Metrics:', final_metrics)
    
    # sample_ds = Subset(trainer.eval_dataset, range(3))
    # pred_output = trainer.predict(sample_ds)
    # pred_logits = pred_output.predictions
    # label_ids = pred_output.label_ids
    
    # if isinstance(pred_logits, tuple):
    #     pred_logits = pred_logits[0]
    # pred_ids = np.argmax(pred_logits, axis=-1)
    
    # pred_texts = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in pred_ids]
    # label_texts = [tokenizer.decode(ids[ids != -100].tolist(), skip_special_tokens=True) for ids in label_ids]
    
    # for i, (pt, lt) in enumerate(zip(pred_texts, label_texts)):
    #     print(f"--- Example {i} ---")
    #     print(f"Predicted: {pt}")
    #     print(f"Correct: {lt}")
    #     print("Answer (pred vs gold):", extract_final_answer(pt), "vs", extract_final_answer(lt))
    
if __name__ == "__main__":
    # main()
    from eval_gsm8k_v5 import batched_generate_eval