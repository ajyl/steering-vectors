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
    ift_state_dict = torch.load(config["ift_state_dict"], map_location=model.device)[
        "state"
    ]
    llama2_state_dict = model.state_dict()

    # embed: model.embed_tokens.weight
    # unembed: lm_head.weight
    # model.norm.weight

    llama2_state_dict["model.embed_tokens.weight"] = ift_state_dict[
        "model.embed_tokens.weight"
    ]
    llama2_state_dict["model.norm.weight"] = ift_state_dict["model.norm.weight"]
    llama2_state_dict["lm_head.weight"] = ift_state_dict["lm_head.weight"]

    model.load_state_dict(llama2_state_dict)
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
            multiplier={
                0: 0.15,
                1: 0.2,
                2: 0.3,
                3: 0.4,
                4: 0.5,
            },
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

    print("Generating for Llama2...")
    llama2_out = _generate_wrapper(model, tokenizer, data, config)

    print("Generating for Llama2 + steer...")
    llama2_w_steer = _hooked_generate_wrapper(
        model, tokenizer, steer_vecs, data, config
    )

    model = stitch_lm_head(model, config)
    print("Generating for Llama2 + ift head...")
    llama2_w_ift_head = _generate_wrapper(model, tokenizer, data, config)
    print("Generating for Llama2 + steer + ift head...")
    llama2_w_steer_ift_head = _hooked_generate_wrapper(
        model, tokenizer, steer_vecs, data, config
    )

    #model, _ = get_model_and_tokenizer(
    #    config["ift_model_path"], config["tokenizer"], device_map="auto"
    #)
    #print("Generating for IFT...")
    #ift_out = _generate_wrapper(model, tokenizer, data, config)

    prompts = data["prompts"]
    num_generate = config["num_generate"]
    with open("outputs.json", "w") as file_p:
        json.dump(
            [
                {
                    "prompts": prompts[idx],
                    "llama2_out": llama2_out[idx],
                    "llama2_w_steer": llama2_w_steer[idx],
                    "llama2_w_ift_head": llama2_w_ift_head[idx],
                    "llama2_w_steer_ift_head": llama2_w_steer_ift_head[idx],
                    # "ift_out": ift_out[idx],
                }
                for idx in range(num_generate)
            ],
            file_p,
            indent=4,
        )

    # for idx in range(num_generate):
    #    print("-----------")
    #    print("| Prompt: |")
    #    print("-----------")
    #    print(f"{prompts[idx]}")

    #    print("-------------")
    #    print("| Original: |")
    #    print("-------------")
    #    print(f"{gpt2_out[idx]}")

    #    print("-----------------------")
    #    print("| GPT2 + IFT LM Head: |")
    #    print("-----------------------")
    #    print(f"{llama2_w_ift_head[idx]}")

    #    print("--------")
    #    print("| IFT: |")
    #    print("--------")
    #    print(f"{ift_out[idx]}")

    #    print("-----------------")
    #    print("| GPT2 + Steer: |")
    #    print("-----------------")
    #    print(f"{gpt2_w_steer[idx]}")

    #    print("-------------------------------")
    #    print("| GPT2 + Steer + IFT LM Head: |")
    #    print("-------------------------------")
    #    print(f"{gpt2_w_steer_ift_head[idx]}")

    breakpoint()


def main():
    """Driver"""
    config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "batch_size": 8,
        "max_prompt_length": 512,
        "max_new_tokens": 40,
        "train_size": 5000,
        "valid_size": 200,
        "cache_dir": "/home/ajyl/ift_mechinterp/data/cache",
        "prompt_token": "[Instruction]",
        "response_token": "[Response]",
        "tokenizer": "meta-llama/Llama-2-7b-hf",
        "steer_timesteps": 10,
        "num_generate": 100,
        "ift_state_dict": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/LATEST/ift_model.pt",
        "ift_model_path": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/llama_ift_LATEST",
        "ckpt_dir": "/home/ajyl/steering-vectors/ckpts/llama2",
    }
    valid_data = load_cached_data(os.path.join(config["cache_dir"], "valid.json"))
    valid_data = build_ift_prompts(valid_data, config)
    valid_data = tokenize_data(valid_data, config)

    steer_vecs = load_steer_vecs(
        os.path.join(config["ckpt_dir"], "steer_vec_series.pt")
    )
    sample(steer_vecs, valid_data, config)


if __name__ == "__main__":
    main()
