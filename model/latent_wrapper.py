import torch
import torch.nn as nn


class LatentTokenWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, latent_token_id: int, start_latent_id: int, end_latent_id: int):
        super().__init__()
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        
        # Fast path: if no latents, defer entirely
        if input_ids is None or not (input_ids == self.latent_token_id).any():
            print("FAST PATH DESPITE LATENT")
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        device = input_ids.device
        
        if labels is not None:
            labels = labels.to(device)
            for tid in self.ignore_token_ids:
                labels[input_ids == tid] = -100

        # device = next(self.base_model.parameters()).device
        # device = self.base_model.get_input_embeddings().weight.device
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
            out = self.base_model.model(
                inputs_embeds=torch.cat(emb_list, dim=1),
                attention_mask=attention_mask[:, : i + 1],
                # labels=labels[:, : i + 1],
                use_cache=False,
                # output_hidden_states=True,
                return_dict=True
            )
            prev_hidden = out.last_hidden_state[:, -1, :]  # requires_grad=True
            # prev_hidden = out.hidden_states[-1][:, -1, :]

        # Full forward with grads on the filled embeds
        if max_last_lat + 1 < T:
            rest = embed(input_ids[:, max_last_lat + 1:])  # [B, T - (max_last_lat+1), H]
            filled = torch.cat([torch.cat(emb_list, dim=1), rest], dim=1)
        else:
            filled = torch.cat(emb_list, dim=1)
            
        out =  self.base_model(inputs_embeds=filled, attention_mask=attention_mask, labels=labels, **kwargs)

        # if labels is not None and hasattr(out, 'logits'):
        #     target_device = labels.device
        #     out.logits = out.logits.to(target_device)
        #     if hasattr(out, 'loss') and out.loss is not None:
        #         out.loss = out.loss.to(target_device)
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