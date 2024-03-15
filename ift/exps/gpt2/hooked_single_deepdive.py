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
from steering_vectors.torch_utils import get_module
from ift.data_utils import (
    get_batch,
    tokenize_data,
    load_cached_data,
    build_ift_prompts,
)


# def build_activation_stack(recorded_activations):
#
#    stack = []
#    # TODO:
#    for layer in range(24):


def sample_ift(data, config):
    """
    Generate with IFT.
    """
    max_length = config["max_prompt_length"]
    max_new_tokens = config["max_new_tokens"]

    ift, tokenizer = get_hooked_model_and_tokenizer(
        config["model"],
        config["tokenizer"],
        model_path=config["ift_model_path"],
        device="cuda:0",
    )
    gpt2, tokenizer = get_hooked_model_and_tokenizer(
        config["model"], config["tokenizer"], device="cuda:1"
    )

    sample = get_batch(data, 18, 1)

    prompt_input_ids = sample["prompt_input_ids"]
    attention_mask = sample["prompt_attention_mask"]

    prompt = sample["prompts"]
    prompt = prompt[0] + " Here is the recipe:"

    # prompt_input_ids_bs = sample["prompt_input_ids"]
    # prompt_shape = prompt_input_ids_bs.shape
    tokenized = tokenizer(
        prompt,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    prompt_input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    prompt_input_ids = torch.cat(
        [prompt_input_ids, torch.tensor([[198, 198]])], dim=1
    )
    attention_mask = torch.cat([attention_mask, torch.tensor([[1, 1]])], dim=1)
    prompt_shape = prompt_input_ids.shape

    ift_device = ift.embed.W_E.device
    gpt2_device = gpt2.embed.W_E.device
    with torch.inference_mode():
        _, ift_cache = ift.run_with_cache(
            prompt_input_ids.to(ift_device),
            attention_mask=attention_mask.to(ift_device),
        )
        _, gpt2_cache = gpt2.run_with_cache(
            prompt_input_ids.to(gpt2_device),
            attention_mask=attention_mask.to(gpt2_device),
        )

    ift_acts = []
    for layer in range(ift.cfg.n_layers):
        ift_acts.append(ift_cache[f"blocks.{layer}.hook_resid_mid"][0, -1, :])
        ift_acts.append(ift_cache[f"blocks.{layer}.hook_resid_post"][0, -1, :])

    ift_acts = torch.stack(ift_acts, dim=0)
    unembed = ift.unembed.W_U
    vocab_proj = einsum(
        "layer d_model, d_model vocab -> layer vocab",
        ift_acts,
        unembed,
    )

    ift_one_probs = vocab_proj.softmax(dim=-1)[:, 16]
    ift_i_probs = vocab_proj.softmax(dim=-1)[:, 40]

    ift_one_rank = (
        vocab_proj.topk(k=ift.cfg.d_vocab).indices == 16
    ).nonzero()[:, 1]
    ift_i_rank = (vocab_proj.topk(k=ift.cfg.d_vocab).indices == 40).nonzero()[
        :, 1
    ]

    gpt2_acts = []
    for layer in range(gpt2.cfg.n_layers):
        gpt2_acts.append(
            gpt2_cache[f"blocks.{layer}.hook_resid_mid"][0, -1, :]
        )
        gpt2_acts.append(
            gpt2_cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        )

    gpt2_acts = torch.stack(gpt2_acts, dim=0)
    unembed = gpt2.unembed.W_U
    vocab_proj = einsum(
        "layer d_model, d_model vocab -> layer vocab",
        gpt2_acts,
        unembed,
    )

    gpt2_one_probs = vocab_proj.softmax(dim=-1)[:, 16]
    gpt2_i_probs = vocab_proj.softmax(dim=-1)[:, 40]

    gpt2_one_rank = (
        vocab_proj.topk(k=gpt2.cfg.d_vocab).indices == 16
    ).nonzero()[:, 1]
    gpt2_i_rank = (
        vocab_proj.topk(k=gpt2.cfg.d_vocab).indices == 40
    ).nonzero()[:, 1]

    plot_data = []
    for layer in range(gpt2_one_probs.shape[0]):

        layer_num = int(layer / 2)
        suffix = "mid"
        if layer_num % 2 == 0:
            suffix = "post"

        plot_data.append(
            {
                "Layer": f"{layer_num}_{suffix}",
                "one_prob": gpt2_one_probs[layer].item(),
                "i_prob": gpt2_i_probs[layer].item(),
                "one_rank": gpt2_one_rank[layer].item(),
                "i_rank": gpt2_i_rank[layer].item(),
                "Model": "GPT2",
            }
        )
        plot_data.append(
            {
                "Layer": f"{layer_num}_{suffix}",
                "one_prob": ift_one_probs[layer].item(),
                "i_prob": ift_i_probs[layer].item(),
                "one_rank": ift_one_rank[layer].item(),
                "i_rank": ift_i_rank[layer].item(),
                "Model": "IFT",
            }
        )

    plot_data = pd.DataFrame(plot_data)

    gs = GridSpec(2, 2)
    fig = plt.figure(figsize=(5, 5))

    for idx in range(4):
        curr_row = idx // 2
        curr_col = idx % 2

        ax = fig.add_subplot(gs[curr_row, curr_col])
        y = {
            0: "i_prob",
            1: "one_prob",
            2: "i_rank",
            3: "one_rank",
        }[idx]
        sns.lineplot(
            data=plot_data,
            x="Layer",
            y=y,
            hue="Model",
            ax=ax
        )

    breakpoint()
    fig.show()

    # "1": 16
    # "I": 40
    # IFT: ['1', '2', 'Ingredients', '6', '3', '*', '##', 'For', '4', '``']
    # IFT: [   16,    17, 41222,    21,    18,     9,  2235,  1890,    19, 15506]

    # GPT2: ['I', '1', '[', 'Ingredients', 'The', 'This', 'You', 'A', '*', 'For']
    # GPT2: [   40,    16,    58, 41222,   464,  1212,  1639,    32,     9,  1890]
    # topk_preds = logits_bsv[:, -1, :].topk(k=10).indices
    # topk_preds_toks = tokenizer.batch_decode(topk_preds)

    breakpoint()
    print("z")

    # print("IFT:")
    # print(output_text)


def sample_steer(steer_vecs, data, config):
    """
    Generate with steering.
    """
    max_new_tokens = config["max_new_tokens"]
    multipliers = config["multipliers"]

    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="auto"
    )

    print("Generating for GPT2 + steer...")
    sample = get_batch(data, 18, 1)
    prompt_input_ids_bs = sample["prompt_input_ids"]
    prompt_shape = prompt_input_ids_bs.shape

    output = hooked_generate(
        model,
        tokenizer,
        steer_vecs,
        sample["prompt_input_ids"].to(model.device),
        max_new_tokens,
        multiplier=multipliers,
        attention_mask=sample["prompt_attention_mask"].to(model.device),
    )
    output_text = tokenizer.batch_decode(
        output[:, prompt_shape[1] :], skip_special_tokens=True
    )
    print("GPT2 + Steer:")
    print(output_text)


def sample_normal(data, config):
    """
    Sample gpt2.
    """
    max_length = config["max_prompt_length"]
    max_new_tokens = config["max_new_tokens"]
    multipliers = config["multipliers"]

    model, tokenizer = get_hooked_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )

    print("Generating for GPT2")
    sample = get_batch(data, 18, 1)

    # Here is the recipe for the Fluorescent Flies:\n\n1 cup (240 ml) water\n1/2 cup (60 ml) DTT blue food coloring\n1/2 cup (
    prompt = sample["prompts"]
    # prompt = prompt[0] + " Here is the recipe for the Flourescent Flies:"
    prompt = prompt[0] + " Here is the recipe:"

    tokenized = tokenizer(
        prompt,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    prompt_input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    prompt_input_ids = torch.cat(
        [prompt_input_ids, torch.tensor([[198, 198]])], dim=1
    )
    attention_mask = torch.cat([attention_mask, torch.tensor([[1, 1]])], dim=1)
    prompt_shape = prompt_input_ids.shape
    model_device = model.embed.W_E.device

    with torch.inference_mode():
        # output = model.generate(
        #    prompt_input_ids.to(model.device),
        #    attention_mask=attention_mask.to(model.device),
        #    do_sample=False,
        #    max_new_tokens=max_new_tokens,
        #    pad_token_id=tokenizer.pad_token_id,
        # )
        logits, cache = model.run_with_cache(
            prompt_input_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
        )

    breakpoint()

    # output_text = tokenizer.batch_decode(
    #    output[:, prompt_shape[1] :], skip_special_tokens=True
    # )

    # prediction = output[:, prompt_shape[1]]
    logits_bsv = logits.logits
    prediction = logits_bsv[:, -1, :].argmax(-1)

    topk_preds = logits_bsv[:, -1, :].topk(k=10).indices
    topk_preds_toks = tokenizer.batch_decode(topk_preds)

    breakpoint()
    print("GPT2:")
    print(output_text)


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
    # sample_steer(steer_vecs, valid_data, config)
    sample_ift(valid_data, config)
    # sample_normal(valid_data, config)


if __name__ == "__main__":
    main()
