import os
import json
from tqdm import tqdm
import torch
from fancy_einsum import einsum

from steering_vectors.train_steering_vector import (
    train_steering_vector,
    train_steering_vector_series,
)
from steering_vectors.record_activations import record_activations
from steering_vectors.steering_vector import SteeringVector, hooked_generate
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs
from ift.data_utils import get_batch, tokenize_data, load_cached_data, build_ift_prompts


def main():
    """Driver"""
    config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "batch_size": 32,
        "max_prompt_length": 512,
        "max_response_length": 64,
        "max_new_tokens": 20,
        "train_size": 5000,
        "valid_size": 200,
        # "data_dir": "/home/repos/ift_mechinterp/data/stack_exchange",
        "cache_dir": "/home/ajyl/ift_mechinterp/data/cache",
        "prompt_token": "[Instruction]",
        "response_token": "[Response]",
        "ift_state_dict": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/LATEST/ift_model.pt",
        "ift_model_path": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/llama_ift_LATEST",
        "tokenizer": "meta-llama/Llama-2-7b-hf",
        "steer_timesteps": 10,
    }
    train_data = load_cached_data(os.path.join(config["cache_dir"], "train.json"))
    train_data = build_ift_prompts(train_data, config)
    train_data = tokenize_data(train_data, config)

    cache_dir = config["cache_dir"]
    pos_sample_filepath = os.path.join(cache_dir, "pos_generations.pt")
    neg_sample_filepath = os.path.join(cache_dir, "neg_generations.pt")

    pos = torch.load(pos_sample_filepath)
    neg = torch.load(neg_sample_filepath)

    contrast_data = [
        {
            "pos_toks": pos[idx].cpu(),
            "neg_toks": neg[idx].cpu(),
            "prompt": train_data["prompts"][idx],
            "prompt_input_ids": train_data["prompt_input_ids"][idx],
        }
        for idx in range(len(train_data["prompts"]))
    ]

    output_filepath = os.path.join(cache_dir, "pos_neg_pairs.pt")
    torch.save(contrast_data, output_filepath)


if __name__ == "__main__":
    main()
