# ponderer/utils/analysis.py
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def measure_latent_distance(activations, layer_index, metric='l2'):
    """
    Measures the distance between activations of a given layer using a specified metric.

    Args:
        activations (dict): A dictionary containing activations.
        layer_index (int): The index of the layer to measure.
        metric (str): The distance metric to use ('l1', 'l2', or 'cosine').

    Returns:
        torch.Tensor: A tensor of normalized distances.
    """
    layer_activations = activations[f'layer_{layer_index}']
    
    activations_except_last = layer_activations[:, :-1]
    activations_except_first = layer_activations[:, 1:]
    
    if metric in ['l1', 'l2']:
        p = 1 if metric == 'l1' else 2
        differences = activations_except_first - activations_except_last
        distances = torch.norm(differences, p=p, dim=2).squeeze(0)
    elif metric == 'cosine':
        cosine_sim = F.cosine_similarity(activations_except_last, activations_except_first, dim=2)
        distances = 1 - cosine_sim.squeeze(0)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'l1', 'l2', 'cosine'.")

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
    plt.ylabel('Normalized Distance', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)