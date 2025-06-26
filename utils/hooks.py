import torch


def setup_hooks(model):
    """Setup hooks on all transformer layers of the model to store their outputs.

    Args:
        model (torch.nn.Module): The model to setup hooks on.

    Returns:
        dict: A dictionary to store activations.
        list: A list of hook handles.
    """
    
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach() # Hook activated upon forward pass
        return hook

    for i, block in enumerate(model.transformer.h):
        hook_handle = block.register_forward_hook(get_activation(f'layer_{i}'))
        hooks.append(hook_handle)

    return activations, hooks


def remove_hooks(hooks):
    """
    Removes all registered hooks.

    Args:
        hooks: A list of hook handles.
    """
    for hook in hooks:
        hook.remove()
        