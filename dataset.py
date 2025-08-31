# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


class TokenizedLatentCollator:
    def __init__(self, tokenizer, start_latent_id, latent_id, end_latent_id, label_pad_token_id=-100, max_seq_length=None):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.start_latent_id = start_latent_id
        self.latent_id = latent_id
        self.end_latent_id = end_latent_id
        self.label_pad_token_id = label_pad_token_id
        self.max_seq_length = max_seq_length
        
    def __call__(self, features):
        # print(features[0])
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        
        def find_first(ids, tid):
            try:
                return ids.index(tid)
            except ValueError:
                return -1
        
        # Left padding
        earliest = [find_first(ids, self.start_latent_id) for ids in input_ids]
        target = max(earliest)
        for i, ids in enumerate(input_ids):
            n_pad = target - earliest[i]
            input_ids[i] = [self.pad_id] * n_pad + ids
            attention_mask[i] = [0] * n_pad + attention_mask[i]
        
        # Right padding
        # max_length = max(len(ids) for ids in input_ids)
        max_length = self.max_seq_length if self.max_seq_length is not None else max(len(ids) for ids in input_ids)
        for i, ids in enumerate(input_ids):
            n_pad = max_length - len(ids)
            input_ids[i] += [self.pad_id] * n_pad
            attention_mask[i] += [0] * n_pad
        
        # Create labels
        labels = []
        for ids, mask in zip(input_ids, attention_mask):
            end_pos = find_first(ids, self.end_latent_id)
            label = []
            for i, tid in enumerate(ids):
                if mask[i] == 0 or (end_pos > 0 and i <= end_pos): # All tokens up to end-latet are inputs/latents
                    label.append(self.label_pad_token_id)
                else:
                    label.append(tid)
            labels.append(label)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
class AnswerOnlyCausalCollator:
    def __init__(self, tokenizer, answer_template="### Answer:", max_seq_length=512, left_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.left_pad_token_id = left_pad_token_id
        self.template_ids = tokenizer(answer_template, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _find_sequence(seq, subseq):
        n, m = len(seq), len(subseq)
        if n == 0 or n < m:
            return None
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return None

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = torch.full_like(input_ids, self.left_pad_token_id)
        
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            attn_len = int(attention_mask[i].sum().item())
            start_idx = self._find_sequence(ids[:attn_len], self.template_ids)
            if start_idx is None:
                continue
            ans_start = start_idx + len(self.template_ids)
            labels[i, ans_start:attn_len] = input_ids[i, ans_start:attn_len]
        
        batch["labels"] = labels
        return batch

class LatentDataloaderCreator:
    """
    DataLoaderCreator for latent-aware training.
    The dataset is split into stages, and the data loader is created for each stage.
    """
    def __init__(self, 
                 ds, 
                 formatting_prompts_func, 
                 tokenizer, 
                 tokenize_fn, 
                 batch_size,
                 data_collator,
                 shuffle=False,
                 num_workers=0,
                 contains_latent=True, 
                 max_latents_max=10,
                 stages=10):
        self.ds = ds
        self.formatting_prompts_func = formatting_prompts_func
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.batch_size = batch_size
        self.data_collator = data_collator
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.contains_latent = contains_latent
        self.max_latents_max = max_latents_max
        self.stages = stages
        
        # Initialize
        self.completed = False
        self.stage = 1
        
    def update_stage(self):
        self.stage += 1
        if self.stage >= self.stages:
            self.completed = True
        
    def create_dataloader(self):
        selected_ds = self.ds.select(range(int(len(self.ds) * (self.stage-1) / self.stages), int(len(self.ds) * (self.stage) / self.stages)))
        max_latents = int(self.max_latents_max * self.stage / self.stages)
        train_text = selected_ds.map(lambda ex: self.formatting_prompts_func(ex, contains_latent=self.contains_latent, max_latents=max_latents), remove_columns=selected_ds.column_names)
        train_tok = train_text.map(lambda ex: self.tokenize_fn(ex, self.tokenizer), remove_columns=train_text.column_names)
        train_loader = DataLoader(train_tok, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.data_collator)
        return train_loader