# ponderer/utils/analysis.py
import torch
import os
import matplotlib.pyplot as plt

def measure_latent_distance(activations, layer_index):
    """Measures the L2 distance between activations of a given layer."""
    layer_activations = activations[f'layer_{layer_index}']
    
    activations_except_last = layer_activations[:, :-1]
    activations_except_first = layer_activations[:, 1:]
    
    differences = activations_except_first - activations_except_last
    distances = torch.norm(differences, p=2, dim=2).squeeze(0)
    
    std_distance = torch.std(distances)
    normalized_distances = distances / (std_distance + 1e-8)
    
    return normalized_distances


def plot_latent_distance(distances_dict, tokens, prompt_length, filename, title, generated_only=False):
    """Plots the latent distance for multiple layers on a single graph."""
    plt.figure(figsize=(18, 8))
    
    start_index = prompt_length - 1 if generated_only else 0
    
    for label, distances in distances_dict.items():
        plot_distances = distances[start_index:]
        plt.plot(plot_distances.cpu().numpy(), label=label, marker='o', linestyle='-', markersize=4)

    plot_tokens = tokens[start_index:]
    tick_labels = [f"{t1.replace('Ġ', ' ')}→{t2.replace('Ġ', ' ')}" for t1, t2 in zip(plot_tokens[:-1], plot_tokens[1:])]
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=90, fontsize=9)
    
    if not generated_only:
        plt.axvline(x=prompt_length - 1.5, color='r', linestyle='--', label='Start of Generation')

    plt.title(title, fontsize=16)
    plt.xlabel('Token Transition', fontsize=12)
    plt.ylabel('Normalized L2 Distance', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)