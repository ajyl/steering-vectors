"""
Module Doc String
"""

import os
import json
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from fancy_einsum import einsum
import torch
import torch.nn.functional as F
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

# %%


def get_acts(model, cache):
    acts = []
    for layer in range(model.cfg.n_layers):
        acts.append(cache[f"blocks.{layer}.hook_resid_mid"][:, -1, :])
        acts.append(cache[f"blocks.{layer}.hook_resid_post"][:, -1, :])
    acts = torch.stack(acts, dim=1)

    return acts


# %%

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

# valid_data = load_cached_data(os.path.join(config["cache_dir"], "valid.json"))
# valid_data = build_ift_prompts(valid_data, config)
# valid_data = tokenize_data(valid_data, config)

with open("repetitive_prompts.json", "r") as file_p:
    data = json.load(file_p)

valid_data = tokenize_data([{"prompt": prompt} for prompt in data], config)


# steer_vecs = load_steer_vecs(
#    os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
# )
# sample_steer(steer_vecs, valid_data, config)
# sample_ift(valid_data, config)
# sample_normal(valid_data, config)

# %%

max_length = config["max_prompt_length"]
max_new_tokens = config["max_new_tokens"]
batch = get_batch(valid_data, 0, 4)

# gpt2, tokenizer = get_model_and_tokenizer(
#    config["model"], config["tokenizer"], device_map="cuda:1"
# )
# with torch.inference_mode():
#    gpt2_out = gpt2.generate(
#        batch["prompt_input_ids"].to(gpt2.device),
#        attention_mask=batch["prompt_attention_mask"].to(gpt2.device),
#        do_sample=False,
#        max_new_tokens=max_new_tokens,
#        pad_token_id=tokenizer.pad_token_id,
#        use_cache=False,
#    )
# prompt_shape = batch["prompt_input_ids"].shape
# gpt2_out_text = tokenizer.batch_decode(
#    gpt2_out[:, prompt_shape[1] :], skip_special_tokens=True
# )
# print(gpt2_out_text)
# breakpoint()
# print("z")

# %%

ift, tokenizer = get_hooked_model_and_tokenizer(
    config["model"],
    config["tokenizer"],
    model_path=config["ift_model_path"],
    device="cuda:0",
)
gpt2, tokenizer = get_hooked_model_and_tokenizer(
    config["model"], config["tokenizer"], device="cuda:1"
)


curr_input_ids = batch["prompt_input_ids"]
curr_attn_mask = batch["prompt_attention_mask"]

gpt2_unembed = gpt2.unembed.W_U
ift_unembed = ift.unembed.W_U
gpt2_device = gpt2_unembed.device
ift_device = ift_unembed.device
batch_size = 4
num_gen = 8

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(batch_size, num_gen)

plot_data = []
for timestep in range(num_gen):
    with torch.inference_mode():
        _, gpt2_cache = gpt2.run_with_cache(
            curr_input_ids.to(gpt2_device),
            attention_mask=curr_attn_mask.to(gpt2_device),
            past_kv_cache=None,
        )
        _, ift_cache = ift.run_with_cache(
            curr_input_ids.to(ift_device),
            attention_mask=curr_attn_mask.to(ift_device),
            past_kv_cache=None,
        )

    gpt2_acts = get_acts(gpt2, gpt2_cache)
    ift_acts = get_acts(ift, ift_cache)

    # [batch, layer, vocab]
    gpt2_vocab_proj = einsum(
        "batch layer d_model, d_model vocab -> batch layer vocab",
        gpt2_acts,
        gpt2_unembed,
    )
    ift_vocab_proj = einsum(
        "batch layer d_model, d_model vocab -> batch layer vocab",
        ift_acts,
        ift_unembed,
    )

    neg_toks = gpt2_vocab_proj[:, -1].topk(k=10001, dim=-1)
    for k in [0, 10, 100, 1000, 10000]:

        neg_tok = neg_toks.indices[:, k]
        neg_tok_idxs = (
            neg_tok.unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, gpt2.cfg.n_layers * 2, 1)
        )

        gpt2_log_probs = gpt2_vocab_proj.log_softmax(dim=-1)
        ift_log_probs = ift_vocab_proj.log_softmax(dim=-1)

        gpt2_neg_logprobs = gpt2_log_probs.gather(dim=-1, index=neg_tok_idxs)
        ift_neg_logprobs = ift_log_probs.gather(
            dim=-1, index=neg_tok_idxs.to(ift_device)
        )

        for sample_idx in range(gpt2_neg_logprobs.shape[0]):
            for layer in range(gpt2.cfg.n_layers * 2):
                layer_num = int(layer / 2)
                suffix = "mid"
                if layer % 2 == 1:
                    suffix = "post"

                if k == 0:
                    plot_data.append(
                        {
                            "Layer": f"{layer_num}_{suffix}",
                            "logprob": ift_neg_logprobs[
                                sample_idx, layer, sample_idx
                            ].item(),
                            "Model": "IFT",
                            "sample_idx": sample_idx,
                            "timestep": timestep,
                            "k": str(k),
                        }
                    )
                    plot_data.append(
                        {
                            "Layer": f"{layer_num}_{suffix}",
                            "logprob": gpt2_neg_logprobs[
                                sample_idx, layer, sample_idx
                            ].item(),
                            "Model": "GPT2",
                            "sample_idx": sample_idx,
                            "timestep": timestep,
                            "k": str(k),
                        }
                    )
                else:
                    plot_data.append(
                        {
                            "Layer": f"{layer_num}_{suffix}",
                            "logprob": ift_neg_logprobs[
                                sample_idx, layer, sample_idx
                            ].item(),
                            "Model": f"IFT_z",
                            "sample_idx": sample_idx,
                            "timestep": timestep,
                            "k": str(k),
                        }
                    )

    curr_input_ids = torch.cat(
        [curr_input_ids.to(neg_tok.device), neg_tok.unsqueeze(-1)], dim=1
    )
    curr_attn_mask = torch.cat(
        [curr_attn_mask, torch.ones((batch_size, 1))], dim=1
    )
    prompt_shape = curr_input_ids.shape


plot_data = pd.DataFrame(plot_data)

#style = {"0": "", "10": ":", "100": ":", "1000": ":", "10000": ":"}
#style={k:v for k,v in zip(plot_data["k"].unique(), [":"] * len(plot_data["k"].unique()))}
#style["0"] = ""
for row in range(batch_size):
    for col in range(num_gen):

        ax = fig.add_subplot(gs[row, col])
        sns.lineplot(
            data=plot_data[
                (plot_data.sample_idx == row) & (plot_data.timestep == col)
            ],
            ax=ax,
            x="Layer",
            y="logprob",
            hue="Model",
            style="k",
            #dashes=style,
        )
        ax.legend([], [], frameon=False)

        if row == 0:
            ax.set_title(f"Timestep {col}")
        if col != 0:
            ax.yaxis.label.set_visible(False)
        ax.xaxis.set_visible(False)


fig.savefig("testing_suppression.pdf")
plt.subplots_adjust(hspace=0.8)


breakpoint()
print("z")
