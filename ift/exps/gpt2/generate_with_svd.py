"""
Module Doc String
"""
import os
import json
from tqdm import tqdm
import torch
from steering_vectors.steering_vector import hooked_generate, SteeringVector
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs
from ift.exps.gpt2.generate_samples import (
    _hooked_generate_wrapper,
    _generate_wrapper,
)
from ift.data_utils import (
    get_batch,
    tokenize_data,
    load_cached_data,
    build_ift_prompts,
)


def get_svd(steer_vecs):
    """
    Try svd...
    """
    steer_vec_per_timestep = []
    for timestep in steer_vecs.keys():

        steer_vec_ld = torch.stack(
            [
                steer_vecs[timestep].layer_activations[layer]
                for layer in range(24)
            ],
            dim=0,
        )
        steer_vec_per_timestep.append(steer_vec_ld)

    steer_vecs_tld = torch.stack(steer_vec_per_timestep, dim=0)

    _steer_vecs = steer_vecs_tld.view((-1, steer_vecs_tld.shape[-1]))
    svd = torch.linalg.svd(_steer_vecs.transpose(0, 1))

    svd_U = svd.U  # [1024, 1024]
    return svd_U


def sample_w_svd(steer_vecs, data, config):
    """
    Generate with steering.
    """
    svd = get_svd(steer_vecs)
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="auto"
    )

    # print("Generating for GPT2...")
    # gpt2_out = _generate_wrapper(model, tokenizer, data, config)

    print("Generating for GPT2 + steer...")

    steer_vec = SteeringVector(
        layer_activations={
            layer: (steer_vecs[4].layer_activations[layer].norm()) * svd[0]
            for layer in range(24)
        }
    )
    svd_steer_vecs = {
        timestep: steer_vec for timestep in range(config["max_new_tokens"])
    }
    svd_steer_vecs[0] = steer_vecs[0]
    gpt2_w_steer = _hooked_generate_wrapper(
        model, tokenizer, svd_steer_vecs, data, config
    )

    # model = stitch_lm_head(model, config)
    # print("Generating for gpt2 + ift head...")
    # gpt2_w_ift_head = _generate_wrapper(model, tokenizer, data, config)
    print("Generating for gpt2 + steer + ift head...")
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

        # print("-------------")
        # print("| Original: |")
        # print("-------------")
        # print(f"{gpt2_out[idx]}")

        # print("-----------------------")
        # print("| GPT2 + IFT LM Head: |")
        # print("-----------------------")
        # print(f"{gpt2_w_ift_head[idx]}")

        print("--------")
        print("| IFT: |")
        print("--------")
        print(f"{ift_out[idx]}")

        print("-----------------")
        print("| GPT2 + Steer: |")
        print("-----------------")
        print(f"{gpt2_w_steer[idx]}")

        # print("-------------------------------")
        # print("| GPT2 + Steer + IFT LM Head: |")
        # print("-------------------------------")
        # print(f"{gpt2_w_steer_ift_head[idx]}")

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
        "num_generate": 10,
        "ckpt_dir": "/home/repos/steering-vectors/ckpts/",
    }

    config["multipliers"] = {
        timestep: 1 for timestep in range(config["steer_timesteps"])
    }
    config["multipliers"][0] = 1

    valid_data = load_cached_data(
        os.path.join(config["cache_dir"], "valid.json")
    )
    valid_data = build_ift_prompts(valid_data, config)
    valid_data = tokenize_data(valid_data, config)

    steer_vecs = load_steer_vecs(
        os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
    )
    sample_w_svd(steer_vecs, valid_data, config)


if __name__ == "__main__":
    main()
