# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
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
        
def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset