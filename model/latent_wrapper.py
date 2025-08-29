import torch
import torch.nn as nn


class LatentTokenWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, latent_token_id: int, start_latent_id: int, end_latent_id: int, pad_token_id: int):
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_token_id = pad_token_id
        self.ignore_token_ids = {latent_token_id, start_latent_id, end_latent_id}

        # Expose HF/Accelerate hints so sharding is preserved
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.hf_device_map = getattr(base_model, "hf_device_map", {})
        self.is_parallelizable = getattr(base_model, "is_parallelizable", False)
        self.model_parallel = getattr(base_model, "model_parallel", False)
        self.device = getattr(base_model, "device", None)

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

    # # Legacy forward pass
    # def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
    #     # Fast path: if no latents, defer entirely
    #     if input_ids is None or not (input_ids == self.latent_token_id).any():
    #         print("FAST PATH DESPITE LATENT")
    #         out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    #         print("COMPLETED FAST PATH")
    #         return out
        
    #     print("LATENT PATH")
    #     device = input_ids.device
        
    #     if labels is not None:
    #         labels = labels.to(device)
    #         for tid in self.ignore_token_ids:
    #             labels[input_ids == tid] = -100

    #     # device = next(self.base_model.parameters()).device
    #     # device = self.base_model.get_input_embeddings().weight.device
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    #     else:
    #         attention_mask = attention_mask.to(device)

    #     embed = self.base_model.get_input_embeddings()
    #     filled = embed(input_ids).clone()
    #     B, T = input_ids.shape

    #     latent_mask = (input_ids == self.latent_token_id)
    #     idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    #     last_lat_per_row = torch.where(latent_mask, idxs, torch.full_like(idxs, -1)).max(dim=1).values
    #     max_last_lat = int(last_lat_per_row.max().item())
    #     prev_hidden = None

    #     emb_list = []
    #     for i in range(max_last_lat + 1):
    #         print(f"COMPUTING LATENT {i} OF {max_last_lat + 1}")
    #         base_emb_i = embed(input_ids[:, i])  # [B, H]
    #         use_latent = (input_ids[:, i] == self.latent_token_id)

    #         if i == 0 and use_latent.any():
    #             bos_id = getattr(self.base_model.config, "bos_token_id", None)
    #             if bos_id is not None:
    #                 bos_emb = embed(torch.full_like(input_ids[:, i], bos_id))
    #                 emb_i = torch.where(use_latent.unsqueeze(-1), bos_emb, base_emb_i)
    #             else:
    #                 emb_i = base_emb_i
    #         elif i > 0 and use_latent.any():
    #             emb_i = torch.where(use_latent.unsqueeze(-1), prev_hidden, base_emb_i)
    #         else:
    #             emb_i = base_emb_i

    #         emb_list.append(emb_i.unsqueeze(1))  # grows the differentiable prefix

    #         # Forward over current prefix to get the last hidden (no cache; keep graph)
    #         out = self.base_model.model(
    #             inputs_embeds=torch.cat(emb_list, dim=1),
    #             attention_mask=attention_mask[:, : i + 1],
    #             # labels=labels[:, : i + 1],
    #             use_cache=False,
    #             # output_hidden_states=True,
    #             return_dict=True
    #         )
    #         prev_hidden = out.last_hidden_state[:, -1, :]  # requires_grad=True
    #         # prev_hidden = out.hidden_states[-1][:, -1, :]

    #     # Full forward with grads on the filled embeds
    #     if max_last_lat + 1 < T:
    #         rest = embed(input_ids[:, max_last_lat + 1:])  # [B, T - (max_last_lat+1), H]
    #         filled = torch.cat([torch.cat(emb_list, dim=1), rest], dim=1)
    #     else:
    #         filled = torch.cat(emb_list, dim=1)
            
    #     out =  self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)

    #     # if labels is not None and hasattr(out, 'logits'):
    #     #     target_device = labels.device
    #     #     out.logits = out.logits.to(target_device)
    #     #     if hasattr(out, 'loss') and out.loss is not None:
    #     #         out.loss = out.loss.to(target_device)
    #     print("COMPLETED LATENT PATH")
    #     return out
    
    # # Distributed Forward Pass
    # def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
    #     # Fast path: if no latents, defer entirely
    #     if input_ids is None:
    #         print("FAST PATH DESPITE LATENT")
    #         out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    #         print("COMPLETED FAST PATH")
    #         return out
        
    #     # For distributed processes
    #     has_latents_local = (input_ids == self.latent_token_id).any()
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         t = torch.tensor([int(has_latents_local)], dtype=torch.int, device=input_ids.device)
    #         torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    #         has_latents_global = bool(t.item())
    #     else:
    #         has_latents_global = bool(has_latents_local)
            
    #     if not has_latents_global:
    #         print("FAST PATH DESPITE LATENT")
    #         out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    #         print("COMPLETED FAST PATH")
    #         return out
        
    #     print("LATENT PATH")
    #     device = input_ids.device
        
    #     if labels is not None:
    #         labels = labels.to(device)
    #         for tid in self.ignore_token_ids:
    #             labels[input_ids == tid] = -100

    #     # device = next(self.base_model.parameters()).device
    #     # device = self.base_model.get_input_embeddings().weight.device
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype, device=device)
    #     else:
    #         attention_mask = attention_mask.to(device)

    #     embed = self.base_model.get_input_embeddings()
    #     filled = embed(input_ids).clone()
    #     B, T = input_ids.shape

    #     latent_mask = (input_ids == self.latent_token_id)
    #     idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    #     last_lat_per_row = torch.where(latent_mask, idxs, torch.full_like(idxs, -1)).max(dim=1).values
    #     max_last_lat_local = int(last_lat_per_row.max().item())
        
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         t_max = torch.tensor([int(max_last_lat_local)], dtype=torch.int, device=input_ids.device)
    #         torch.distributed.all_reduce(t_max, op=torch.distributed.ReduceOp.MAX)
    #         max_last_lat_global = int(t_max.item())
    #     else:
    #         max_last_lat_global = max_last_lat_local
            
    #     pad_id = getattr(self.base_model.config, "pad_token_id", None)
    #     if pad_id is not None:
    #         pad_emb = embed(torch.full((B,), pad_id, dtype=input_ids.dtype, device=device))
    #     else:
    #         pad_emb = torch.zeros((B, embed.embedding_dim), dtype=embed.weight.dtype, device=device)
        
    #     prev_hidden = None
    #     emb_list = []
    #     print("INPUT ID SHAPE:", input_ids.shape)
    #     for i in range(max_last_lat_global + 1):
    #         print(f"COMPUTING LATENT {i} OF {max_last_lat_local + 1}")
    #         in_range = i <T
    #         if in_range:
    #             base_emb_i = embed(input_ids[:, i])  # [B, H]
    #             use_latent = (input_ids[:, i] == self.latent_token_id)
    #         else:
    #             base_emb_i = pad_emb
    #             use_latent = torch.zeros_like(input_ids[:, i], dtype=torch.bool, device=device)

    #         if i == 0 and use_latent.any():
    #             bos_id = getattr(self.base_model.config, "bos_token_id", None)
    #             if bos_id is not None:
    #                 bos_emb = embed(torch.full_like(input_ids[:, i], bos_id))
    #                 emb_i = torch.where(use_latent.unsqueeze(-1), bos_emb, base_emb_i)
    #             else:
    #                 emb_i = base_emb_i
    #         elif i > 0 and use_latent.any():
    #             emb_i = torch.where(use_latent.unsqueeze(-1), prev_hidden, base_emb_i)
    #         else:
    #             emb_i = base_emb_i

    #         emb_list.append(emb_i.unsqueeze(1))  # grows the differentiable prefix

    #         # Forward over current prefix to get the last hidden (no cache; keep graph)
    #         # print(f"ENTERING BASE MODEL FORWARD PASS {i} OF {max_last_lat + 1}")
    #         # out = self.base_model.model(
    #         #     inputs_embeds=torch.cat(emb_list, dim=1),
    #         #     attention_mask=attention_mask[:, : i + 1],
    #         #     # labels=labels[:, : i + 1],
    #         #     use_cache=False,
    #         #     # output_hidden_states=True,
    #         #     return_dict=True
    #         # )
    #         # prev_hidden = out.last_hidden_state[:, -1, :]  # requires_grad=True
            
    #         if i+1 <= T:
    #             am_step = attention_mask[:, : i + 1]
    #         else:
    #             pad_cols = torch.zeros((B, i+1-T), dtype=attention_mask.dtype, device=device)
    #             am_step = torch.cat([attention_mask, pad_cols], dim=1)
            
    #         out = self.base_model(
    #             inputs_embeds=torch.cat(emb_list, dim=1),
    #             attention_mask=am_step,
    #             use_cache=False,
    #             output_hidden_states=True,
    #             return_dict=True
    #         )
    #         prev_hidden = out.hidden_states[-1][:, -1, :]

    #     # Full forward with grads on the filled embeds
    #     if max_last_lat_local == -1:
    #         filled = embed(input_ids)
    #     elif max_last_lat_local + 1 < T:
    #         prefix = torch.cat(emb_list[:max_last_lat_local + 1], dim=1)
    #         rest = embed(input_ids[:, max_last_lat_local + 1:])
    #         filled = torch.cat([prefix, rest], dim=1)
    #     else:
    #         filled = torch.cat(emb_list, dim=1)
            
    #     out = self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
        
    #     # start = max_last_lat_global + 1
    #     # if start < T:
    #     #     rest = embed(input_ids[:, max_last_lat_local + 1:])  # [B, T - (max_last_lat+1), H]
    #     #     filled = torch.cat([torch.cat(emb_list, dim=1), rest], dim=1)
    #     # else:
    #     #     filled = torch.cat(emb_list, dim=1)
            
    #     # out =  self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)

    #     # if labels is not None and hasattr(out, 'logits'):
    #     #     target_device = labels.device
    #     #     out.logits = out.logits.to(target_device)
    #     #     if hasattr(out, 'loss') and out.loss is not None:
    #     #         out.loss = out.loss.to(target_device)
    #     print("COMPLETED LATENT PATH")
    #     return out
    
    # Distributed Forward Pass v2 - Apply maximum max_n_latents forward passes by indexing latent positions
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Fast path: if no latents, defer entirely
        if input_ids is None:
            print("FAST PATH DESPITE LATENT")
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            print("COMPLETED FAST PATH")
            return out
        
        # For distributed processes        
        has_latents_local = (input_ids == self.latent_token_id).any()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([int(has_latents_local)], dtype=torch.int, device=input_ids.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            has_latents_global = bool(t.item())
        else:
            has_latents_global = bool(has_latents_local)
            
        if not has_latents_global:
            print("FAST PATH DESPITE LATENT")
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            print("COMPLETED FAST PATH")
            return out
        
        print("LATENT PATH")
        device = input_ids.device
        
        if labels is not None:
            labels = labels.to(device)
            for tid in self.ignore_token_ids:
                labels[input_ids == tid] = -100

        # device = next(self.base_model.parameters()).device
        # device = self.base_model.get_input_embeddings().weight.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype, device=device)
        else:
            attention_mask = attention_mask.to(device)

        embed = self.base_model.get_input_embeddings()
        filled = embed(input_ids).clone()
        B, T = input_ids.shape

        latent_mask = (input_ids == self.latent_token_id)
        start_latent_mask = (input_ids == self.start_latent_id)
        end_latent_mask = (input_ids == self.end_latent_id)
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        first_lat_per_row = torch.where(start_latent_mask, idxs, torch.full_like(idxs, T)).min(dim=1).values
        last_lat_per_row = torch.where(end_latent_mask, idxs, torch.full_like(idxs, -1)).max(dim=1).values
        min_first_lat_local = int(first_lat_per_row.min().item())+1
        max_last_lat_local = int(last_lat_per_row.max().item())-1
        diff_lat_local = max_last_lat_local - min_first_lat_local + 1 # Number of latents to compute
        
        print(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, min_first_lat_local, max_last_lat_local, diff_lat_local, T)
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t_diff = torch.tensor([int(diff_lat_local)], dtype=torch.int, device=input_ids.device)
            torch.distributed.all_reduce(t_diff, op=torch.distributed.ReduceOp.MAX)
            diff_lat_global = int(t_diff.item())
        else:
            diff_lat_global = diff_lat_local
        
        # Extend input, attention mask, and labels for extended answers (distributed)
        if min_first_lat_local + diff_lat_global > T:
            print("IMPORTANT2", min_first_lat_local, diff_lat_local, diff_lat_global, T)
            print("EXTRA:", min_first_lat_local + diff_lat_global - T, type(min_first_lat_local + diff_lat_global - T))
            pad_emb = embed(torch.full((B, min_first_lat_local + diff_lat_global - T,), 
                                       self.pad_token_id, 
                                       dtype=input_ids.dtype, 
                                       device=device))
            filled = torch.cat([filled, pad_emb], dim=1)
            attention_mask = torch.cat([attention_mask, 
                                        torch.zeros((B, min_first_lat_local + diff_lat_global - T), 
                                                   dtype=attention_mask.dtype, 
                                                   device=device)],
                                      dim=1)
            labels = torch.cat([labels, 
                                torch.full((B, min_first_lat_local + diff_lat_global - T), -100, dtype=labels.dtype, device=device)],
                                dim=1)
        
        print("INPUT ID SHAPE:", input_ids.shape)
        for i in range(min_first_lat_local, min_first_lat_local + diff_lat_global):
            print(f"\n COMPUTING LATENT {i}/{min_first_lat_local+diff_lat_global-1} (Tot: {i-min_first_lat_local} / {diff_lat_global})")
            
            # Compute prev_hidden from prefix to add to position i
            assert i > 0, "Latent position 0 is not supported"
            out = self.base_model(
                inputs_embeds = filled[:, : i, :],
                attention_mask = attention_mask[:, : i],
                use_cache = False,
                output_hidden_states = True,
                return_dict = True
            )
            prev_hidden = out.hidden_states[-1][:, -1, :]

            # Fill in the latent token(s) at position i
            if latent_mask[:, i].any():
                filled[:, i, :] = torch.where(latent_mask[:, i].unsqueeze(-1), prev_hidden, filled[:, i, :])

        # out = self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
        # print("COMPLETED LATENT PATH")
        
        print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: filled.shape={filled.shape}, attention_mask.shape={attention_mask.shape}")
        if labels is not None:
            print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: labels.shape={labels.shape}")
        out = self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
        print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: Completed final base_model forward, out.logits.shape={out.logits.shape if hasattr(out, 'logits') else 'N/A'}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            print(f"Rank {torch.distributed.get_rank()}: Passed barrier after final forward")

        return out
    
    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        n_latents = kwargs.pop("n_latents", None)
        max_latent_length = kwargs.pop("max_latent_length", 128)
        end_latent_threshold = kwargs.pop("end_latent_threshold", 0.9)
        
        if input_ids is None:
            raise ValueError("Generate expects 'input_ids' in kwargs")
        
        device = input_ids.device
        B = input_ids.size(0)
        
        embed = self.base_model.get_input_embeddings()
        lm_head = self.base_model.get_output_embeddings()
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)
        
        # 1. Append start-of-latent token id
        start_ids = torch.full((B, 1), self.start_latent_id, dtype=input_ids.dtype, device=device)
        input_ids_ext = torch.cat([input_ids, start_ids], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
        
        inputs_embeds = embed(input_ids_ext)
        
        # 2. If type(fixed_latents) == int -> append <|latent|> * fixed_latents
        if isinstance(n_latents, int):
            with torch.no_grad():
                for _ in range(n_latents):
                    out = self.base_model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True
                    )
                    prev_hidden = out.last_hidden_state[:, -1, :]
                    inputs_embeds = torch.cat([inputs_embeds, prev_hidden.unsqueeze(1)], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)],
                        dim=1
                    )
                    
            end_ids = torch.full((B, 1), self.end_latent_id, dtype=input_ids.dtype, device=device)
            end_emb = embed(end_ids)
            inputs_embeds = torch.cat([inputs_embeds, end_emb], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)],
                dim=1
            )
            
            # Generate as usual
            kwargs.pop("input_ids", None)
            kwargs.pop("attention_mask", None)
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['attention_mask'] = attention_mask
            
            return self.base_model.generate(*args, **kwargs)
        
        # If no fixed_latents specified, let model freestyle
        else:
            alive = torch.ones(B, dtype=torch.bool, device=device)
            zero_embed = torch.zeros((B, embed.embedding_dim), dtype=inputs_embeds.dtype, device=device)
            
            with torch.no_grad():
                for i in range(max_latent_length):
                    out = self.base_model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True
                    )
                    prev_hidden = out.last_hidden_state[:, -1, :]
                    logits = lm_head(prev_hidden)
                    probs = torch.softmax(logits, dim=-1)
                    end_p = probs[:, self.end_latent_id]
                    
                    newly_end = alive & (end_p > end_latent_threshold)
                    end_emb = embed(torch.full((B,), self.end_latent_id, dtype=input_ids.dtype, device=device))
                    
                    next_step_embed = torch.where(
                        newly_end.unsqueeze(-1),
                        end_emb,
                        torch.where(alive.unsqueeze(-1), prev_hidden, zero_embed)
                    ) # [B, H]
                    
                    inputs_embeds = torch.cat([inputs_embeds, next_step_embed.unsqueeze(1)], dim=1)
                    next_mask_col = alive.long().unsqueeze(1)
                    attention_mask = torch.cat([attention_mask, next_mask_col], dim=1)
                    
                    alive = alive & (~newly_end)
                    if not alive.any():
                        break
                    
            # End any alive latent sequences
            if alive.any():
                end_emb = embed(torch.full((B,), self.end_latent_id, dtype=input_ids.dtype, device=device))
                append_emb = torch.where(alive.unsqueeze(-1), end_emb, zero_embed)
                inputs_embeds = torch.cat([inputs_embeds, append_emb.unsqueeze(1)], dim=1)
                mask_end = alive.long().unsqueeze(1)
                attention_mask = torch.cat([attention_mask, mask_end], dim=1)
            
            kwargs.pop("input_ids", None)
            kwargs.pop("attention_mask", None)
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['attention_mask'] = attention_mask
            
            return self.base_model.generate(*args, **kwargs)
        
        
class LatentTokenWrapperNoDP(nn.Module):
    def __init__(self, base_model: nn.Module, latent_token_id: int, start_latent_id: int, end_latent_id: int, pad_token_id: int, verbose: bool = False):
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_token_id = pad_token_id
        self.ignore_token_ids = {latent_token_id, start_latent_id, end_latent_id}
        self.verbose = verbose

        # Expose HF/Accelerate hints so sharding is preserved
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.hf_device_map = getattr(base_model, "hf_device_map", {})
        self.is_parallelizable = getattr(base_model, "is_parallelizable", False)
        self.model_parallel = getattr(base_model, "model_parallel", False)
        self.device = getattr(base_model, "device", None)

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
    
    def _vprint(self, *args, rank_only: bool = True, **kwargs):
        if not getattr(self, "verbose", False):
            return
        if rank_only and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank != 0:
                return
            print(f"[rank {rank}]", *args, **kwargs)
        else:
            print(*args, **kwargs)
    
    # Distributed Forward Pass v2 - Apply maximum max_n_latents forward passes by indexing latent positions
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Fast path: if no latents, defer entirely
        if input_ids is None:
            self._vprint("FAST PATH DESPITE LATENT")
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            self._vprint("COMPLETED FAST PATH")
            return out
        
        # For distributed processes        
        has_latents = (input_ids == self.latent_token_id).any()
            
        if not has_latents:
            self._vprint("FAST PATH DESPITE LATENT")
            out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            self._vprint("COMPLETED FAST PATH")
            return out
        
        self._vprint("LATENT PATH")
        # device = input_ids.device
        embed = self.base_model.get_input_embeddings()
        device = embed.weight.device
        input_ids = input_ids.to(device)
        
        if labels is not None:
            labels = labels.to(device)
            for tid in self.ignore_token_ids:
                labels[input_ids == tid] = -100

        # device = next(self.base_model.parameters()).device
        # device = self.base_model.get_input_embeddings().weight.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype) # DS, device=device
        else:
            attention_mask = attention_mask.to(device)

        filled = embed(input_ids).clone().to(device)
        B, T = input_ids.shape

        latent_mask = (input_ids == self.latent_token_id)
        start_latent_mask = (input_ids == self.start_latent_id)
        end_latent_mask = (input_ids == self.end_latent_id)
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        first_lat_per_row = torch.where(start_latent_mask, idxs, torch.full_like(idxs, T)).min(dim=1).values
        last_lat_per_row = torch.where(end_latent_mask, idxs, torch.full_like(idxs, -1)).max(dim=1).values
        min_first_lat = int(first_lat_per_row.min().item())+1
        max_last_lat = int(last_lat_per_row.max().item())-1
        diff_lat = max_last_lat - min_first_lat + 1 # Number of latents to compute
        
        self._vprint(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, min_first_lat, max_last_lat, diff_lat, T)
        
        # Extend input, attention mask, and labels for extended answers (distributed)
        if min_first_lat + diff_lat > T:
            self._vprint("IMPORTANT2", min_first_lat, diff_lat, diff_lat, T)
            self._vprint("EXTRA:", min_first_lat + diff_lat - T, type(min_first_lat + diff_lat - T))
            pad_emb = embed(torch.full((B, min_first_lat + diff_lat - T,), 
                                       self.pad_token_id, 
                                       dtype=input_ids.dtype, 
                                       device=device))
            filled = torch.cat([filled, pad_emb], dim=1)
            attention_mask = torch.cat([attention_mask, 
                                        torch.zeros((B, min_first_lat + diff_lat - T), 
                                                   dtype=attention_mask.dtype, 
                                                   device=device)],
                                      dim=1)
            labels = torch.cat([labels, 
                                torch.full((B, min_first_lat + diff_lat - T), -100, dtype=labels.dtype, device=device)],
                                dim=1)
        
        self._vprint("INPUT ID SHAPE:", input_ids.shape)
        
        def _prefix_forward(emb, am, **kwargs):
            out = self.base_model.model(
                inputs_embeds = emb,
                attention_mask = am,
                use_cache = False,
                return_dict = True
            )
            return out.last_hidden_state[:, -1, :]
        
        
        for i in range(min_first_lat, min_first_lat + diff_lat):
            self._vprint(f"\n COMPUTING LATENT {i}/{min_first_lat+diff_lat-1} (Tot: {i-min_first_lat+1} / {diff_lat})")
            
            # Compute prev_hidden from prefix to add to position i
            assert i > 0, "Latent position 0 is not supported"
            
            # out = self.base_model(
            #     inputs_embeds = filled[:, : i, :],
            #     attention_mask = attention_mask[:, : i],
            #     use_cache = False,
            #     # output_hidden_states = True,
            #     return_dict = True
            # )
            # prev_hidden = out.hidden_states[-1][:, -1, :]
            
            prev_hidden = torch.utils.checkpoint.checkpoint(
                _prefix_forward,
                filled[:, :i, :].clone().requires_grad_(True),
                attention_mask[:, :i],
            ).to(filled.device)

            # Fill in the latent token(s) at position i
            if latent_mask[:, i].any():
                filled[:, i, :] = torch.where(latent_mask[:, i].unsqueeze(-1), prev_hidden, filled[:, i, :])

        # out = self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
        # print("COMPLETED LATENT PATH")
        
        out = self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)
        
        return out
    
    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        n_latents = kwargs.pop("n_latents", None)
        max_latent_length = kwargs.pop("max_latent_length", 128)
        end_latent_threshold = kwargs.pop("end_latent_threshold", 0.9)
        
        if input_ids is None:
            raise ValueError("Generate expects 'input_ids' in kwargs")
        
        device = input_ids.device
        B = input_ids.size(0)
        
        embed = self.base_model.get_input_embeddings()
        lm_head = self.base_model.get_output_embeddings()
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)
        
        # 1. Append start-of-latent token id
        start_latent_ids = torch.full((B, 1), self.start_latent_id, dtype=input_ids.dtype, device=device)
        input_ids_ext = torch.cat([input_ids, start_latent_ids], dim=1)
        am = torch.cat([attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
        
        inputs_embs_prefix = embed(input_ids_ext)
        
        # 2. If type(fixed_latents) == int -> append <|latent|> * fixed_latents
        with torch.no_grad():
            out = self.base_model.model(
                inputs_embeds=inputs_embs_prefix,
                attention_mask=am,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            prev_hidden = out.last_hidden_state[:, -1, :]
            
            gen_embs = []
            
            # Fixed n_latent steps
            if isinstance(n_latents, int):
                ones_col = torch.ones((B, 1), dtype=am.dtype, device=device)
                for _ in range(n_latents):
                    step_emb = prev_hidden
                    gen_embs.append(step_emb.unsqueeze(1))
                    am = torch.cat([am, ones_col], dim=1)
                    
                    step_out = self.base_model.model(
                        inputs_embeds=step_emb.unsqueeze(1),
                        attention_mask=am,
                        use_cache=True,
                        past_key_values=past_kv,
                        return_dict=True,
                    )
                    past_kv = step_out.past_key_values
                    prev_hidden = step_out.last_hidden_state[:, -1, :]
                    
                end_latent_ids = torch.full((B,1), self.end_latent_id, dtype=input_ids.dtype, device=device)
                gen_embs.append(embed(end_latent_ids))
                am = torch.cat([am, ones_col], dim=1)
                
            # Unconstrained latent steps with early stop
            else:
                alive = torch.ones(B, dtype=torch.bool, device=device)
                zero_emb = torch.zeros((B, embed.embedding_dim), dtype=inputs_embs_prefix.dtype, device=device)
                end_lat_emb = embed(torch.full((B,), self.end_latent_id, dtype=input_ids.dtype, device=device))
                
                for _ in range(max_latent_length):
                    logits = lm_head(prev_hidden) # [B, V]
                    probs = torch.softmax(logits, dim=-1)
                    end_p = probs[:, self.end_latent_id] # [B]
                    
                    newly_end = alive & (end_p > end_latent_threshold)
                    
                    # End lat emb where latent ended, latent embedding where latent is alive, 0 otherwise (attention masked)
                    next_step_emb = torch.where(
                        newly_end.unsqueeze(-1), # [B, 1]
                        end_lat_emb,
                        torch.where(alive.unsqueeze(-1), prev_hidden, zero_emb)
                    )                 
                    
                    gen_embs.append(next_step_emb.unsqueeze(1)) # [B, 1, H]
                    am = torch.cat([am, alive.long().unsqueeze(1)], dim=1) # [B, T + 1]
                    
                    step_out = self.base_model.model(
                        inputs_embeds = next_step_emb.unsqueeze(1),
                        attention_mask = am,
                        use_cache = True,
                        past_key_values = past_kv,
                        return_dict = True,
                    )
                    past_kv = step_out.past_key_values
                    prev_hidden = step_out.last_hidden_state[:, -1, :]
                    
                    alive = alive & (~newly_end)
                    if not alive.any():
                        break
                
                if alive.any():
                    append_emb = torch.where(alive.unsqueeze(-1), end_lat_emb, zero_emb)
                    gen_embs.append(append_emb.unsqueeze(1))
                    am = torch.cat([am, alive.long().unsqueeze(1)], dim=1)
                    
        if gen_embs:
            gen_embs_cat = torch.cat(gen_embs, dim=1)
            inputs_embs_final = torch.cat([inputs_embs_prefix, gen_embs_cat], dim=1)
        else:
            inputs_embs_final = inputs_embs_prefix
                    
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs['inputs_embeds'] = inputs_embs_final
        kwargs['attention_mask'] = am
        
        return self.base_model.generate(*args, **kwargs)