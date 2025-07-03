import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from utils.distance import measure_latent_distance, plot_latent_distance
from generation import generate_and_monitor_standard


def main():
    with open('data/gsm_test.json', 'r') as f:
        gsm_data = json.load(f)

    question_index = 0
    question = gsm_data[question_index]['question']

    model_id = 'gpt2'
    model_path = 'results/gpt2_gsm8k'
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.load_state_dict(torch.load(model_path))
        
    tokenizer.pad_token = tokenizer.eos_token

    # The model needs to be in eval mode for hooks to work correctly without affecting gradients
    model.eval()

    # Example Usage
    prompt = f"Q: {question}\nA: Let's think step by step."

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    prompt_length = input_ids.shape[1]
    
    generated_text, activations, final_input_ids = generate_and_monitor_standard(model, tokenizer, input_ids)
    all_tokens = tokenizer.convert_ids_to_tokens(final_input_ids[0])
    
    num_layers = len(activations)
    print('Number of layers', num_layers)

    print("Generated Text:")
    print(generated_text)
    print("\n" + "="*50 + "\n")
    
    # Calculate distances for various layers
    metric = 'l2'
    distances_start = measure_latent_distance(activations, 0, metric=metric)
    first_quartile, second_quartile, third_quartile, fourth_quartile = num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers-1
    distances_first_quartile = measure_latent_distance(activations, first_quartile, metric=metric)
    distances_second_quartile = measure_latent_distance(activations, second_quartile, metric=metric)
    distances_third_quartile = measure_latent_distance(activations, third_quartile, metric=metric)
    distances_end = measure_latent_distance(activations, fourth_quartile, metric=metric)
    
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
        f'Normalized {metric.upper()} Latent Distance Between Consecutive Tokens'
    )
    print("Plot saved to results/latent_distance.png")

    # Plot the distances for the generated part only
    plot_latent_distance(
        distances_to_plot, 
        all_tokens, 
        prompt_length,
        'results/latent_distance_generated_only.png',
        f'Normalized {metric.upper()} Latent Distance Between Consecutive Tokens (Generated Part Only)',
        generated_only=True
    )
    print("Plot of generated part saved to results/latent_distance_generated_only.png")

    
if __name__ == '__main__':
    main()