import torch
from utils.hooks import setup_hooks, remove_hooks

def generate_and_monitor_standard(model, tokenizer, input_ids, max_length=50):
    """Generates text autoregressively while monitoring activations."""
    activations, hooks = setup_hooks(model)

    # Generate text token by token
    for i in range(max_length):
        outputs = model(input_ids)
        
        # Get the next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the generated token to the input_ids
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode the full generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    remove_hooks(hooks)
    
    return generated_text, activations, input_ids 