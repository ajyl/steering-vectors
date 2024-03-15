"""
Experiments for steering.
"""

import os
import json
from tqdm import tqdm
import torch
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs
from steering_vectors.steering_vector import hooked_generate
from ift.data_utils import (
    get_batch,
    tokenize_data,
    load_cached_data,
    build_ift_prompts,
)


def stitch_lm_head(model, config):
    """
    Return GPT2 with lm head swapped out.
    """
    ift_state_dict = torch.load(config["ift_state_dict"])["state"]
    gpt2_state_dict = model.state_dict()
    gpt2_state_dict["transformer.wte.weight"] = ift_state_dict[
        "transformer.wte.weight"
    ]
    gpt2_state_dict["transformer.ln_f.weight"] = ift_state_dict[
        "transformer.ln_f.weight"
    ]
    gpt2_state_dict["transformer.ln_f.bias"] = ift_state_dict[
        "transformer.ln_f.bias"
    ]
    gpt2_state_dict["lm_head.weight"] = ift_state_dict["lm_head.weight"]

    model.load_state_dict(gpt2_state_dict)
    return model


def _generate_wrapper(model, tokenizer, data, config):
    """
    Wrapper around generate.
    """
    num_generate = config["num_generate"]
    batch_size = config["batch_size"]
    max_new_tokens = config["max_new_tokens"]

    all_outputs = []
    for idx in tqdm(range(0, num_generate, batch_size)):
        batch = get_batch(data, idx, batch_size)

        prompt_input_ids_bs = batch["prompt_input_ids"]
        prompt_shape = prompt_input_ids_bs.shape
        with torch.inference_mode():
            output = model.generate(
                batch["prompt_input_ids"].to(model.device),
                attention_mask=batch["prompt_attention_mask"].to(model.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                # use_cache=False,
            )

        output_text = tokenizer.batch_decode(
            output[:, prompt_shape[1] :], skip_special_tokens=True
        )
        all_outputs.extend(output_text)
    return all_outputs


def _hooked_generate_wrapper(model, tokenizer, steer_vecs, data, config):
    """
    Wrapper around hooked generate.
    """
    num_generate = config["num_generate"]
    batch_size = config["batch_size"]
    max_new_tokens = config["max_new_tokens"]

    all_outputs = []

    multipliers = config["multipliers"]
    for idx in tqdm(range(0, num_generate, batch_size)):
        batch = get_batch(data, idx, batch_size)

        prompt_input_ids_bs = batch["prompt_input_ids"]
        prompt_shape = prompt_input_ids_bs.shape

        output = hooked_generate(
            model,
            tokenizer,
            steer_vecs,
            batch["prompt_input_ids"].to(model.device),
            max_new_tokens,
            multiplier=multipliers,
            attention_mask=batch["prompt_attention_mask"].to(model.device),
        )

        hooked_text = tokenizer.batch_decode(
            output[:, prompt_shape[1] :], skip_special_tokens=True
        )
        all_outputs.extend(hooked_text)
    return all_outputs


def sample(steer_vecs, data, config):
    """
    Generate with steering.
    """
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="auto"
    )

    print("Generating for GPT2...")
    gpt2_out = _generate_wrapper(model, tokenizer, data, config)

    # print("Generating for GPT2 + steer...")
    # gpt2_w_steer = _hooked_generate_wrapper(
    #    model, tokenizer, steer_vecs, data, config
    # )

    # model = stitch_lm_head(model, config)
    # print("Generating for gpt2 + ift head...")
    # gpt2_w_ift_head = _generate_wrapper(model, tokenizer, data, config)
    # print("Generating for gpt2 + steer + ift head...")
    # gpt2_w_steer_ift_head = _hooked_generate_wrapper(
    #    model, tokenizer, steer_vecs, data, config
    # )

    model, _ = get_model_and_tokenizer(
        config["ift_model_path"], config["tokenizer"], device_map="auto"
    )
    print("Generating for IFT...")
    ift_out = _generate_wrapper(model, tokenizer, data, config)

    prompts = data["prompts"]
    num_generate = config["num_generate"]
    # with open("outputs.json", "w") as file_p:
    #    json.dump(
    #        [
    #            {
    #                "prompts": prompts[idx],
    #                "gpt2_out": gpt2_out[idx],
    #                "gpt2_w_steer": gpt2_w_steer[idx],
    #                # "gpt2_w_ift_head": gpt2_w_ift_head[idx],
    #                "gpt2_w_steer_ift_head": gpt2_w_steer_ift_head[idx],
    #                # "ift_out": ift_out[idx],
    #            }
    #            for idx in range(num_generate)
    #        ],
    #        file_p,
    #        indent=4,
    #    )

    for idx in range(num_generate):
        print("-----------")
        print("| Prompt: |")
        print("-----------")
        print(f"{prompts[idx]}")

        print("-------------")
        print("| Original (GPT2): |")
        print("-------------")
        print(f"{gpt2_out[idx]}")

        # print("-----------------------")
        # print("| GPT2 + IFT LM Head: |")
        # print("-----------------------")
        # print(f"{gpt2_w_ift_head[idx]}")

        print("--------")
        print("| IFT: |")
        print("--------")
        print(f"{ift_out[idx]}")

        # print("-----------------")
        # print("| GPT2 + Steer: |")
        # print("-----------------")
        # print(f"{gpt2_w_steer[idx]}")

        # print("-------------------------------")
        # print("| GPT2 + Steer + IFT LM Head: |")
        # print("-------------------------------")
        # print(f"{gpt2_w_steer_ift_head[idx]}")


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

    steer_vecs = load_steer_vecs(
        os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
    )

    # for timestep in steer_vecs.keys():
    #    #for layer in range(12, 24):
    #    for layer in range(12):
    #        steer_vecs[timestep].layer_activations.pop(layer)

    sample(steer_vecs, valid_data, config)


if __name__ == "__main__":
    main()
