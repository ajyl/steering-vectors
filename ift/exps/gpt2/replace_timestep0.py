"""
Deep dive on single samples
"""

import os
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from fancy_einsum import einsum
import torch
from ift.interp_utils import (
    get_model_and_tokenizer,
    load_steer_vecs,
    get_hooked_model_and_tokenizer,
)
from steering_vectors.steering_vector import hooked_generate
from steering_vectors.record_activations import record_activations
from steering_vectors.torch_utils import get_module, untuple_tensor
from ift.data_utils import (
    get_batch,
    tokenize_data,
    load_cached_data,
    build_ift_prompts,
)


def _add_hook(patch):
    """
    z
    """

    def hook_fn(_module, _inputs):
        # [batch, timestep, d_model]
        orig_inputs = untuple_tensor(_inputs)
        orig_inputs[:, 0, :] = patch
        return _inputs

    return hook_fn


def swap_hooked_generate(
    ift, gpt2, tokenizer, steer_vecs, batch, gen_timesteps, **model_kwargs
):
    input_ids = batch["prompt_input_ids"].to(gpt2.device)
    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id]).to(
        input_ids.device
    )
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    layers = [f"transformer.h.{layer}" for layer in range(24)]
    for timestep in range(gen_timesteps):

        handle = None
        steer_vec = steer_vecs.get(timestep)
        if steer_vec is not None:
            handle = steer_vec.patch_activations(
                gpt2, multiplier=1
            )

        model_inputs = gpt2.prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )
        model_inputs["use_cache"] = False

        with torch.inference_mode():
            with record_activations(ift, layer_nums=None) as record:
                output = ift.forward(
                    batch["prompt_input_ids"].to(ift.device),
                    attention_mask=batch["prompt_attention_mask"].to(
                        ift.device
                    ),
                )

                # [batch, layers, d_model]
                stacked = torch.stack(
                    [record[layer][0][:, 0, :] for layer in layers], dim=1
                )

        hooks = []
        for layer_idx, layer_name in enumerate(layers):
            _patch = stacked[:, layer_idx, :]
            module = get_module(gpt2, layer_name)
            hook = module.register_forward_pre_hook(_add_hook(_patch))
            hooks.append(hook)

        with torch.inference_mode():
            outputs = gpt2.forward(**model_inputs)

        next_token_logits = outputs.logits[:, -1, :].clone()

        if timestep == 0:
            # " The"
            next_token_logits[:, 383] = -1e10

        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = gpt2._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1)
            .ne(eos_token_id_tensor.unsqueeze(1))
            .prod(dim=0)
        )
        for hook in hooks:
            hook.remove()

        if handle is not None:
            handle.remove()

        if unfinished_sequences.max() == 0:
            break

    return input_ids


def swap(data, config):
    """
    Swap experiment.
    """
    steering_vectors = load_steer_vecs(
       os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
    )
    batch_size = config["batch_size"]
    max_new_tokens = config["max_new_tokens"]

    gpt2, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map=0
    )
    ift, _ = get_model_and_tokenizer(
        config["ift_model_path"], config["tokenizer"], device_map=1
    )
    for idx in tqdm(range(0, len(data["prompts"]), batch_size)):
        batch = get_batch(data, idx, batch_size)

        batch = get_batch(data, 18, 1)

        #prompt_shape = prompt_input_ids_bs.shape
        prompt = batch["prompts"][0] + " Here is the recipe:"

        tokenized = tokenizer(
            prompt,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        prompt_input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        batch["prompt_input_ids"] = torch.cat(
            [prompt_input_ids, torch.tensor([[198, 198]])], dim=1
        )
        batch["prompt_attention_mask"] = torch.cat([attention_mask, torch.tensor([[1, 1]])], dim=1)
        prompt_shape = batch["prompt_input_ids"].shape

        with torch.inference_mode():
            # testing = gpt2.forward(
            #    batch["prompt_input_ids"].to(gpt2.device),
            #    attention_mask=batch["prompt_attention_mask"].to(gpt2.device),
            # )
            testing = gpt2.generate(
                batch["prompt_input_ids"].to(gpt2.device),
                attention_mask=batch["prompt_attention_mask"].to(gpt2.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                # use_cache=False,
            )
        before_text = tokenizer.batch_decode(
            testing[:, prompt_shape[1] :], skip_special_tokens=True
        )

        swapped_gen = swap_hooked_generate(
            ift,
            gpt2,
            tokenizer,
            steering_vectors,
            batch,
            20,
            attention_mask=batch["prompt_attention_mask"].to(gpt2.device),
        )

        swapped_gen_text = tokenizer.batch_decode(
            swapped_gen[:, prompt_shape[1] :], skip_special_tokens=True
        )
        for _idx, z in enumerate(swapped_gen_text):
            print("Before:")
            print(before_text[_idx])
            print("After:")
            print(z)

        breakpoint()


def main():
    """ Driver """
    config = {
        "model": "gpt2-medium",
        "batch_size": 16,
        "max_prompt_length": 512,
        "max_new_tokens": 40,
        "train_size": 5000,
        "valid_size": 200,
        "cache_dir": "/home/repos/ift_mechinterp/data/cache",
        "prompt_token": "[Instruction]",
        "response_token": "[Response]",
        "ift_state_dict": "/home/repos/ift_mechinterp/finetune/.cache/andrew/gpt2_ift/LATEST/ift_model.pt",
        "ift_model_path": "/home/repos/ift_mechinterp/ift_gpt2_latest",
        "tokenizer": "gpt2-medium",
        "steer_timesteps": 20,
        "num_generate": 20,
        "ckpt_dir": "/home/repos/steering-vectors/ckpts/",
    }

    config["multipliers"] = {
        timestep: 1 for timestep in range(config["steer_timesteps"])
    }
    config["multipliers"][0] = 0.1

    valid_data = load_cached_data(
        os.path.join(config["cache_dir"], "valid.json")
    )
    valid_data = build_ift_prompts(valid_data, config)
    valid_data = tokenize_data(valid_data, config)

    swap(valid_data, config)

    # steer_vecs = load_steer_vecs(
    #    os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
    # )
    # sample_steer(steer_vecs, valid_data, config)
    # sample_ift(valid_data, config)
    # sample_normal(valid_data, config)


if __name__ == "__main__":
    main()
