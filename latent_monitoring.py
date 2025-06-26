import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import matplotlib.pyplot as plt
import json
from utils.hooks import setup_hooks, remove_hooks

def generate_latent_activations(model, tokenizer, input_ids, max_length=50):
    """
    Generates a response to a prompt and monitors the latent activations
    at each step of the generation process.

    Args:
        model: The model for monitoring.
        tokenizer: The tokenizer.
        prompt (str): The input question or prompt.
        max_length (int): The maximum length of the generated sequence.

    Returns:
        A tuple containing the generated text and a dictionary of activations
        at each generation step.
    """
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

def measure_latent_distance(activations, layer_index):
    """
    Measures the L2 distance between activations over the step outputs of a given layer.

    Args:
        activations (list): A list of dictionaries containing activations at each step.
        layer_index (int): The index of the layer to measure the distance.

    Returns:
        float: A list of L2 distances between activations over the step outputs of the given layer.
    """
    # Get the activations for the given layer
    layer_activations = activations[f'layer_{layer_index}']

    # Calculate the L2 distance between activations over the step outputs of the given layer
    # Select all activations except the last one, and all except the first one
    activations_except_last = layer_activations[:, :-1]
    activations_except_first = layer_activations[:, 1:]
    
    # Calculate the element-wise difference (vector from token step t to t+1)
    differences = activations_except_first - activations_except_last # [1, seq_len, emb_dim] -> [1, seq_len-1, emb_dim]
    
    # Calculate the L2 norm (Euclidean distance) for each time step's activation vector
    distances = torch.norm(differences, p=2, dim=2).squeeze(0) # [1, seq_len-1, emb_dim] -> [seq_len-1]
    
    # Normalize the distances
    # mean_distance = torch.mean(distances)
    std_distance = torch.std(distances)
    
    # Avoid division by zero
    # normalized_distances = (distances - mean_distance) / (std_distance + 1e-8)
    normalized_distances = distances / (std_distance + 1e-8)
    
    return normalized_distances

def plot_latent_distance(distances_dict, tokens, prompt_length, filename, title, generated_only=False):
    """
    Plots the latent distance for multiple layers on a single graph.

    Args:
        distances_dict (dict): A dictionary of distances, with labels as keys.
        tokens (list): A list of all tokens (prompt + generated).
        prompt_length (int): The length of the initial prompt.
        filename (str): The filename to save the plot to.
        title (str): The title for the plot.
        generated_only (bool): If True, plot only the generated part of the sequence.
    """
    plt.figure(figsize=(18, 8))
    
    start_index = prompt_length - 1 if generated_only else 0
    
    for label, distances in distances_dict.items():
        plot_distances = distances[start_index:]
        plt.plot(plot_distances.cpu().numpy(), label=label, marker='o', linestyle='-', markersize=4)

    # Create labels for the x-axis showing the transition between tokens
    plot_tokens = tokens[start_index:]
    tick_labels = [f"{tok1.replace('Ġ', ' ')}→{tok2.replace('Ġ', ' ')}" for tok1, tok2 in zip(plot_tokens[:-1], plot_tokens[1:])]
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=90, fontsize=9)
    
    # Add a vertical line to distinguish prompt from generation if plotting the full sequence
    if not generated_only:
        plt.axvline(x=prompt_length - 1.5, color='r', linestyle='--', label='Start of Generation')

    plt.title(title, fontsize=16)
    plt.xlabel('Token Transition', fontsize=12)
    plt.ylabel('Normalized L2 Distance', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout to make room for rotated labels
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename)
    
if __name__ == '__main__':
    # Load GSM8K dataset
    try:
        with open('data/gsm_test.json', 'r') as f:
            gsm_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'data/gsm_test.json' not found.")
        print("Please make sure the GSM8K test set is available at the specified path.")
        exit()

    # Select a question from the dataset
    question_index = 0  # You can change this index to explore different questions
    if question_index < len(gsm_data):
        question = gsm_data[question_index]['question']
    else:
        print(f"Error: question_index {question_index} is out of bounds for the dataset with {len(gsm_data)} questions.")
        exit()

    # Load pre-trained model and tokenizer
    model_id = 'gpt2'
    # model_path = 'results/gpt2_gsm8k'
    model_path = 'results/gpt2_gsm8k_latent'
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.load_state_dict(torch.load(model_path))
    except (OSError, FileNotFoundError):
        print(f"Error: Base model '{model_id}' not found, or weight file not found at '{model_path}'.")
        print("Please ensure the base model is available and the fine-tuned model is saved at the correct path.")
        exit()
        
    tokenizer.pad_token = tokenizer.eos_token

    # The model needs to be in eval mode for hooks to work correctly without affecting gradients
    model.eval()

    # Example Usage
    prompt = f"Q: {question}\nA: Let's think step by step."

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    prompt_length = input_ids.shape[1]
    
    generated_text, activations, final_input_ids = generate_latent_activations(model, tokenizer, input_ids)
    all_tokens = tokenizer.convert_ids_to_tokens(final_input_ids[0])
    
    num_layers = len(activations)
    print('Number of layers', num_layers)

    print("Generated Text:")
    print(generated_text)
    print("\n" + "="*50 + "\n")
    
    # Calculate distances for various layers
    distances_start = measure_latent_distance(activations, 0)
    first_quartile, second_quartile, third_quartile, fourth_quartile = num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers-1
    distances_first_quartile = measure_latent_distance(activations, first_quartile)
    distances_second_quartile = measure_latent_distance(activations, second_quartile)
    distances_third_quartile = measure_latent_distance(activations, third_quartile)
    distances_end = measure_latent_distance(activations, fourth_quartile)
    
    # Combine distances into a dictionary for plotting
    distances_to_plot = {
        'Layer 0 (Start)': distances_start,
        f'Layer {first_quartile} (1st Quartile)': distances_first_quartile,
        f'Layer {second_quartile} (2nd Quartile)': distances_second_quartile,
        f'Layer {third_quartile} (3rd Quartile)': distances_third_quartile,
        f'Layer {fourth_quartile} (End)': distances_end,
    }

    # Plot the distances
    plot_latent_distance(
        distances_to_plot, 
        all_tokens, 
        prompt_length,
        'results/latent_distance.png',
        'Normalized Latent Distance Between Consecutive Tokens'
    )
    print("Plot saved to results/latent_distance.png")

    # Plot the distances for the generated part only
    plot_latent_distance(
        distances_to_plot, 
        all_tokens, 
        prompt_length,
        'results/latent_distance_generated_only.png',
        'Normalized Latent Distance Between Consecutive Tokens (Generated Part Only)',
        generated_only=True
    )
    print("Plot of generated part saved to results/latent_distance_generated_only.png")