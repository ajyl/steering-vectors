"""
Utility functions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors.steering_vector import SteeringVector


def get_model_and_tokenizer(
    model_name_or_path, tokenizer_name, device_map="auto"
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if model_name_or_path.startswith("gpt2"):
        tokenizer.padding_side = "left"
    return model, tokenizer


def load_steer_vecs(filepath):
    _steer_vecs = torch.load(filepath)
    return {
        timestep: SteeringVector(
            layer_activations=vec["layer_activations"],
            layer_type=vec["layer_type"],
        )
        for timestep, vec in _steer_vecs.items()
    }
