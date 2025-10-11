"""
LoRA-related utility functions.
"""


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    """Create parameter optimization configuration."""
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }


def negate_params(name, negation):
    """
    We have to do this if we are co-training with LoRA.
    This ensures that parameter groups aren't duplicated.
    """
    if negation is None:
        return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False
