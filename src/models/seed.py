import random
import numpy as np
import torch

def set_seed(seed=0):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten_config(config):
    """
    Convert a sweep-style config with nested parameter dicts
    into a flat key:value dict.
    """
    params = config.get("parameters", {})
    flat = {}
    for k, v in params.items():
        if "value" in v:
            flat[k] = v["value"]
        elif "values" in v:
            flat[k] = v["values"][0]
        elif "min" in v and "max" in v:
            flat[k] = v["min"]
        else:
            flat[k] = v
    return flat