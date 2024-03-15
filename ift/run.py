"""
Testing...
"""
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



def generate(data, config, pos_or_neg):
    """
    pos_or_neg: {"positive", "negative"}
    """
    model_name_or_path = (
        config["model"]
        if pos_or_neg == "negative"
        else config["ift_model_path"]
    )
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path, config["tokenizer"]
    )

    batch_size = config["batch_size"]
    max_new_tokens = config["steer_timesteps"]

    len_data = len(data["prompts"])
    generations = []
    for idx in tqdm(range(0, len_data, batch_size)):
        batch = get_batch(data, idx, batch_size)
        with torch.inference_mode():
            output = model.generate(
                batch["prompt_input_ids"].to(model.device),
                attention_mask=batch["prompt_attention_mask"].to(model.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        generations.append(output)

    return torch.cat(generations, dim=0).cpu()


def generate_series(data, pos, config):
    """
    Generate sequentially based on pos.
    """
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )

    batch_size = config["batch_size"]
    steer_timesteps = config["steer_timesteps"]

    len_data = len(data["prompts"])
    generations = []

    for idx in tqdm(range(0, len_data, batch_size)):
        batch = get_batch(data, idx, batch_size)
        curr_batch_size = batch["prompt_input_ids"].shape[0]
        prompt_shape = batch["prompt_input_ids"].shape[1]

        outputs_per_timestep = []
        for timestep in range(steer_timesteps):
            prompt_w_ift_bs = pos[
                idx : idx + batch_size,
                : prompt_shape + timestep,
            ]

            with torch.inference_mode():
                output_bs = model.generate(
                    prompt_w_ift_bs.to(model.device),
                    attention_mask=(
                        prompt_w_ift_bs != tokenizer.pad_token_id
                    ).to(model.device),
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            assert output_bs.shape == (
                curr_batch_size,
                prompt_w_ift_bs.shape[1] + 1,
            )

            pad_size = (prompt_shape + steer_timesteps) - output_bs.shape[1]
            padding_bs = (
                torch.ones((output_bs.shape[0], pad_size)).to(output_bs.device)
                * tokenizer.pad_token_id
            ).long()
            output_bs = torch.cat([output_bs, padding_bs], dim=1)
            assert output_bs.shape == (
                curr_batch_size,
                prompt_shape + steer_timesteps,
            )

            outputs_per_timestep.append(output_bs)

        # [batch, steer_timesteps, prompt+steer_timesteps]
        outputs_bts = torch.stack(outputs_per_timestep, dim=1).cpu()
        assert outputs_bts.shape == (
            curr_batch_size,
            steer_timesteps,
            prompt_shape + steer_timesteps,
        )
        generations.append(outputs_bts)

    return torch.cat(generations, dim=0)


def make_pos_neg_pair(data, config, cache_path=None):
    """
    Make positive and negative contrastive pairs.
    """
    # data["prompts"] = data["prompts"][:config["batch_size"]]
    data_size = len(data["prompts"])
    print("Generating positive samples.")
    pos_toks_Ds = generate(data, config, "positive").cpu()
    print("Generating negative samples.")
    neg_toks_Dts = generate_series(data, pos_toks_Ds, config).cpu()

    assert pos_toks_Ds.shape == (
        data_size,
        config["max_prompt_length"] + config["steer_timesteps"],
    )
    assert neg_toks_Dts.shape == (
        data_size,
        config["steer_timesteps"],
        config["max_prompt_length"] + config["steer_timesteps"],
    )

    contrast_data = [
        {
            "pos_toks": pos_toks_Ds[idx].cpu(),
            "neg_toks": neg_toks_Dts[idx].cpu(),
            "prompt": data["prompts"][idx],
            "prompt_input_ids": data["prompt_input_ids"][idx],
        }
        for idx in range(data_size)
    ]

    if cache_path is not None:
        torch.save(contrast_data, cache_path)

    return contrast_data


def load_pos_neg_pair_from_file(filepath):
    """
    Load from file.
    """
    return torch.load(filepath)


def get_steer_vector(data, config):
    """
    Get contrastive vector.
    """
    contrast_data = make_pos_neg_pair(
        data,
        config,
        cache_path=os.path.join(config["cache_dir"], "pos_neg_pairs_longer.pt"),
    )
    contrast_data = load_pos_neg_pair_from_file(
        os.path.join(config["cache_dir"], "pos_neg_pairs_longer.pt")
    )

    gpt2, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )
    steering_vector = train_steering_vector_series(
        gpt2,
        tokenizer,
        contrast_data,
        config["steer_timesteps"],
        layers=None,
        read_token_index=-1,
        move_to_cpu=True,
        batch_size=config["batch_size"],
        show_progress=True,
        prompt_length=config["max_prompt_length"],
    )
    return steering_vector


def steer(steer_vec, data, config):
    """
    Run steering.
    """
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )
    batch_size = config["batch_size"]

    for scale in [0.2]:
        with steer_vec.apply(model, multiplier=scale, min_token_index=None):

            for idx in tqdm(range(0, len(data["prompts"]), batch_size)):
                batch = get_batch(data, idx, batch_size)
                with torch.inference_mode():
                    output = model.generate(
                        batch["prompt_input_ids"].to(model.device),
                        attention_mask=batch["prompt_attention_mask"].to(
                            model.device
                        ),
                        do_sample=False,
                        max_new_tokens=10,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    print(
                        json.dumps(
                            tokenizer.batch_decode(
                                output, skip_special_tokens=True
                            ),
                            indent=4,
                        )
                    )
                    break


def testing_hooked_generate(steer_vecs, data, config):
    """
    Hmm
    """

    batch_size = config["batch_size"]
    max_new_tokens = config["max_new_tokens"]
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map=0
    )
    ift, _ = get_model_and_tokenizer(
        config["ift_model_path"], config["tokenizer"], device_map=1
    )
    for idx in tqdm(range(0, len(data["prompts"]), batch_size)):
        batch = get_batch(data, idx, batch_size)

        prompt_input_ids_bs = batch["prompt_input_ids"]
        prompt_shape = prompt_input_ids_bs.shape
        prompts = batch["prompts"]

        with torch.inference_mode():

            original = model.generate(
                batch["prompt_input_ids"].to(model.device),
                attention_mask=batch["prompt_attention_mask"].to(model.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

            original_resp = tokenizer.batch_decode(
                original[:, prompt_shape[1] :], skip_special_tokens=True
            )

            output = hooked_generate(
                model,
                tokenizer,
                steer_vecs,
                batch["prompt_input_ids"].to(model.device),
                20,
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

            ift_output = ift.generate(
                batch["prompt_input_ids"].to(ift.device),
                attention_mask=batch["prompt_attention_mask"].to(ift.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            ift_output_text = tokenizer.batch_decode(
                ift_output[:, prompt_shape[1] :], skip_special_tokens=True
            )

            for idx in range(len(prompts)):
                print("-----------")
                print("| Prompt: |")
                print("-----------")
                print(f"{prompts[idx]}")

                print("-------------")
                print("| Original: |")
                print("-------------")
                print(f"{original_resp[idx]}")

                print("-------------")
                print("| IFT: |")
                print("-------------")
                print(f"{ift_output_text[idx]}")

                print("-------------")
                print("| Hooked: |")
                print("-------------")
                print(f"{hooked_text[idx]}")

            breakpoint()


def save_steer_vecs(steer_vecs, filepath):
    _steer_vecs = {
        timestep: {
            "layer_activations": vec.layer_activations,
            "layer_type": vec.layer_type,
        }
        for timestep, vec in steer_vecs.items()
    }
    torch.save(_steer_vecs, filepath)


def deepdive(steer_vecs, data, config):
    """
    Hmm..
    """
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )
    steer_vec = steer_vecs[0]

    unembed_vd = model.lm_head.weight

    batch_size = config["batch_size"]
    with steer_vec.apply(model, multiplier=0.15):
        for idx in tqdm(range(0, len(data["prompts"]), batch_size)):
            batch = get_batch(data, idx, batch_size)

            with record_activations(
                model, steer_vec.layer_type, layer_nums=None
            ) as record:
                output = model.forward(
                    batch["prompt_input_ids"].to(model.device),
                    attention_mask=batch["prompt_attention_mask"].to(
                        model.device
                    ),
                )

                # record:
                # {
                #   layer: list[seq]
                #       where list[i] = tensor[batch, seq, d_model]
                # }
                # [batch, layers, d_model]
                acts_bld = torch.stack(
                    [
                        record[layer][0][:, -1, :].cpu()
                        for layer in range(24)  # TODO
                    ],
                    dim=1,
                )
                acts_bld = model.transformer.ln_f(acts_bld)
                dot_prods_blv = einsum(
                    "batch layers d_model, d_model vocab -> batch layers vocab",
                    acts_bld.to(unembed_vd.device),
                    unembed_vd.transpose(0, 1),
                )
                breakpoint()

                logit_lens_blk = dot_prods_blv.topk(k=10, dim=2).indices

            logits_bv = output.logits[:, -1, :]
            topk_bk = logits_bv.topk(k=10, dim=1).indices

            topk_tokens = tokenizer.batch_decode(topk_bk)

            print(topk_tokens)
            breakpoint()
            print("z")


def main():
    """ Driver """
    config = {
        "model": "gpt2-medium",
        "batch_size": 8,
        "max_prompt_length": 512,
        "max_response_length": 64,
        "max_new_tokens": 20,
        "train_size": 5000,
        "valid_size": 200,
        "data_dir": "/home/repos/ift_mechinterp/data/stack_exchange",
        "add_generations": False,
        "use_cached_generations": False,
        "cache_dir": "/home/repos/ift_mechinterp/data/cache",
        "prompt_token": "[Instruction]",
        "response_token": "[Response]",
        "ift_state_dict": "/home/repos/ift_mechinterp/finetune/.cache/andrew/gpt2_ift/LATEST/ift_model.pt",
        "ift_model_path": "/home/repos/ift_mechinterp/ift_gpt2_latest",
        "tokenizer": "gpt2-medium",
        "steer_timesteps": 20,
    }
    train_data = load_cached_data(
        os.path.join(config["cache_dir"], "train.json")
    )
    valid_data = load_cached_data(
        os.path.join(config["cache_dir"], "valid.json")
    )

    train_data = build_ift_prompts(train_data, config)
    valid_data = build_ift_prompts(valid_data, config)

    train_data = tokenize_data(train_data, config)
    valid_data = tokenize_data(valid_data, config)

    #steer_vecs = get_steer_vector(train_data, config)

    #save_steer_vecs(steer_vecs, "steer_vec_series_longer.pt")
    # steer_vec.dump("steer_vec_series.pt")

    # steer_vec = SteeringVector.load_from_file("steer_vec_series.pt")
    steer_vecs = load_steer_vecs("steer_vec_series_longer.pt")
    # steer(steer_vec, valid_data, config)

    deepdive(steer_vecs, valid_data, config)
    #testing_hooked_generate(steer_vecs, valid_data, config)


if __name__ == "__main__":
    main()
