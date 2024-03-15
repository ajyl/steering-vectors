# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# 29: ">"
# 314: " I"
# 1101: "'m"
# 2111: " trying"
# 284: " to"
# 5879: " prove"
# 25: ":"
# 611: " if"
# 39280: " $\\"

# %%

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

# Notes:
# " [Response]" --> [58, 31077, 60] (indices 145:148)
#

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

valid_data = load_cached_data(os.path.join(config["cache_dir"], "valid.json"))
valid_data = build_ift_prompts(valid_data, config)
valid_data = tokenize_data(valid_data, config)

# with open("repetitive_prompts.json", "r") as file_p:
#    data = json.load(file_p)
#
# valid_data = tokenize_data([{"prompt": prompt} for prompt in data], config)


# steer_vecs = load_steer_vecs(
#    os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
# )
# sample_steer(steer_vecs, valid_data, config)
# sample_ift(valid_data, config)
# sample_normal(valid_data, config)

# %%

max_length = config["max_prompt_length"]
max_new_tokens = config["max_new_tokens"]

ift, tokenizer = get_hooked_model_and_tokenizer(
    config["model"],
    config["tokenizer"],
    model_path=config["ift_model_path"],
    device="cuda:0",
)
gpt2, tokenizer = get_model_and_tokenizer(
    config["model"], config["tokenizer"], device_map="cuda:1"
)

# %%

sample = get_batch(valid_data, 1, 1)

prompt_input_ids = sample["prompt_input_ids"]
attention_mask = sample["prompt_attention_mask"]

prompt = sample["prompts"]
prompt = prompt[0] + " > "

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


with torch.inference_mode():
    gpt2_out = gpt2.generate(
        prompt_input_ids.to(gpt2.device),
        attention_mask=attention_mask.to(gpt2.device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    gpt2_out_text = tokenizer.batch_decode(
        gpt2_out[:, prompt_shape[1] :], skip_special_tokens=True
    )

gpt2_out = gpt2_out[:, prompt_shape[1] :]

gpt2, tokenizer = get_hooked_model_and_tokenizer(
    config["model"], config["tokenizer"], device="cuda:1"
)

# %%

ift_device = ift.embed.W_E.device
gpt2_device = gpt2.embed.W_E.device
# tok_ids = [29, 314, 1101, 2111, 284, 5879, 25, 611, 39280]
tok_ids = gpt2_out
prompt_shape = prompt_input_ids.shape
print(tok_ids)


curr_input_ids = prompt_input_ids
curr_attn_mask = attention_mask
for tok_id in tok_ids[0][:12]:

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

    gpt2_acts = []
    for layer in range(gpt2.cfg.n_layers):
        gpt2_acts.append(
            gpt2_cache[f"blocks.{layer}.hook_resid_mid"][0, -1, :]
        )
        gpt2_acts.append(
            gpt2_cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        )
    gpt2_acts = torch.stack(gpt2_acts, dim=0)
    gpt2_unembed = gpt2.unembed.W_U
    gpt2_vocab_proj = einsum(
        "layer d_model, d_model vocab -> layer vocab",
        gpt2_acts,
        gpt2_unembed,
    )
    ift_acts = []
    for layer in range(gpt2.cfg.n_layers):
        ift_acts.append(ift_cache[f"blocks.{layer}.hook_resid_mid"][0, -1, :])
        ift_acts.append(ift_cache[f"blocks.{layer}.hook_resid_post"][0, -1, :])
    ift_acts = torch.stack(ift_acts, dim=0)
    ift_unembed = ift.unembed.W_U
    ift_vocab_proj = einsum(
        "layer d_model, d_model vocab -> layer vocab",
        ift_acts,
        ift_unembed,
    )

    neg_tok = gpt2_vocab_proj[-1].argmax(dim=-1)

    gpt2_neg_logprobs = gpt2_vocab_proj.log_softmax(dim=-1)[:, neg_tok]
    gpt2_neg_rank = (
        gpt2_vocab_proj.topk(k=gpt2.cfg.d_vocab).indices == neg_tok
    ).nonzero()[:, 1]
    ift_neg_logprobs = ift_vocab_proj.log_softmax(dim=-1)[:, neg_tok]
    ift_neg_rank = (
        ift_vocab_proj.topk(k=ift.cfg.d_vocab).indices
        == neg_tok.to(ift_device)
    ).nonzero()[:, 1]

    plot_data = []
    for layer in range(gpt2.cfg.n_layers * 2):

        layer_num = int(layer / 2)
        suffix = "mid"
        if layer % 2 == 1:
            suffix = "post"

        plot_data.append(
            {
                "Layer": f"{layer_num}_{suffix}",
                "neg_logprob": ift_neg_logprobs[layer].item(),
                # "neg_rank": ift_neg_rank[layer].item(),
                "Model": "IFT",
            }
        )
        plot_data.append(
            {
                "Layer": f"{layer_num}_{suffix}",
                "neg_logprob": gpt2_neg_logprobs[layer].item(),
                # "neg_rank": gpt2_neg_rank[layer].item(),
                "Model": "GPT2",
            }
        )

    neg_toks = gpt2_vocab_proj[-1].topk(k=10001, dim=-1)
    for k in [10, 100, 100, 10000]:
        neg_tok = neg_toks.indices[k]
        gpt2_log_probs = gpt2_vocab_proj.log_softmax(dim=-1)[:, neg_tok]
        ift_log_probs = ift_vocab_proj.log_softmax(dim=-1)[:, neg_tok]
        for layer in range(gpt2.cfg.n_layers * 2):

            layer_num = int(layer / 2)
            suffix = "mid"
            if layer % 2 == 1:
                suffix = "post"

            plot_data.append(
                {
                    "Layer": f"{layer_num}_{suffix}",
                    "neg_logprob": ift_log_probs[layer].item(),
                    # "neg_rank": ift_neg_rank[layer].item(),
                    "Model": f"IFT_{k}",
                    "k": k,
                }
            )
            plot_data.append(
                {
                    "Layer": f"{layer_num}_{suffix}",
                    "neg_logprob": gpt2_log_probs[layer].item(),
                    # "neg_rank": gpt2_neg_rank[layer].item(),
                    "Model": f"GPT2_{k}",
                    "k": k,
                }
            )

    plot_data = pd.DataFrame(plot_data)

    print(f"Neg tok: {neg_tok}")
    print(f"Tok id: {tok_id}")
    # px.line(gpt2_i_probs.detach().cpu(), labels={"x": labels})
    # px.line(plot_data, x="Layer", y="neg_prob", color="Model").show(None)
    px.line(plot_data, x="Layer", y="neg_logprob", color="Model").show(None)
    #px.line(plot_data, x="Layer", y="neg_logprob", color="Model").show(None)
    # px.line(plot_data, x="Layer", y="neg_rank", color="Model").show(None)
    # px.line(plot_data, x="Layer", y="pos_prob", color="Model").show(None)
    # px.line(plot_data, x="Layer", y="pos_logprob", color="Model").show(None)
    # px.line(plot_data, x="Layer", y="pos_rank", color="Model").show(None)
    print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")

    curr_input_ids = torch.cat(
        [curr_input_ids, torch.tensor([[tok_id.item()]])], dim=1
    )
    curr_attn_mask = torch.cat([curr_attn_mask, torch.tensor([[1]])], dim=1)
    prompt_shape = curr_input_ids.shape

    breakpoint()

# %%

plot_data = []
for layer in range(gpt2_pos_probs.shape[0]):

    layer_num = int(layer / 2)
    suffix = "mid"
    if layer % 2 == 1:
        suffix = "post"

    plot_data.append(
        {
            "Layer": f"{layer_num}_{suffix}",
            "pos_prob": gpt2_pos_probs[layer].item(),
            "pos_logprob": gpt2_pos_logprobs[layer].item(),
            "neg_prob": gpt2_neg_probs[layer].item(),
            "neg_logprob": gpt2_neg_logprobs[layer].item(),
            "pos_rank": gpt2_pos_rank[layer].item(),
            "neg_rank": gpt2_neg_rank[layer].item(),
            "Model": "GPT2",
        }
    )
    plot_data.append(
        {
            "Layer": f"{layer_num}_{suffix}",
            "pos_prob": ift_pos_probs[layer].item(),
            "pos_logprob": ift_pos_logprobs[layer].item(),
            "neg_prob": ift_neg_probs[layer].item(),
            "neg_logprob": ift_neg_logprobs[layer].item(),
            "pos_rank": ift_pos_rank[layer].item(),
            "neg_rank": ift_neg_rank[layer].item(),
            "Model": "IFT",
        }
    )

plot_data = pd.DataFrame(plot_data)

# %%

# px.line(gpt2_i_probs.detach().cpu(), labels={"x": labels})
px.line(plot_data, x="Layer", y="neg_prob", color="Model").show(None)
px.line(plot_data, x="Layer", y="neg_logprob", color="Model").show(None)
px.line(plot_data, x="Layer", y="neg_rank", color="Model").show(None)
px.line(plot_data, x="Layer", y="pos_prob", color="Model").show(None)
px.line(plot_data, x="Layer", y="pos_logprob", color="Model").show(None)
px.line(plot_data, x="Layer", y="pos_rank", color="Model").show(None)

# %%


with torch.inference_mode():
    ift_cache.compute_head_results()
    gpt2_cache.compute_head_results()


# %%

layer = 9

# [batch, heads, seq, seq]
gpt2_attn_pattern = gpt2_cache[f"blocks.{layer}.attn.hook_pattern"]
gpt2_attn_pattern[0, :, -1, :].max(dim=1)
ift_attn_pattern = ift_cache[f"blocks.{layer}.attn.hook_pattern"]
ift_attn_pattern[0, :, -1, :].max(dim=1)

breakpoint()

# [batch seq heads d_model] --> [heads d_model]
gpt2_decomp_heads = gpt2_cache[f"blocks.{layer}.attn.hook_result"][
    0, -1, ...
].clone()
ift_decomp_heads = ift_cache[f"blocks.{layer}.attn.hook_result"][
    0, -1, ...
].clone()


gpt2_heads_proj_vocab = (
    einsum(
        "heads d_model, d_model vocab -> heads vocab",
        gpt2_decomp_heads,
        unembed.to(gpt2_device),
    )
    .topk(k=10, dim=1)
    .indices
)
ift_heads_proj_vocab = (
    einsum(
        "heads d_model, d_model vocab -> heads vocab",
        ift_decomp_heads,
        unembed.to(ift_device),
    )
    .topk(k=10, dim=1)
    .indices
)

# d_model
next_tok_vec = unembed[:, neg_tok]


# (at layer 15)
# tokenizer.batch_decode(gpt2_heads_proj_vocab[2]) --> "my, mine, myself, gonna, propositions, me, tomorrow, I, future, hypothetical"

breakpoint()


ift_head_outs = torch.stack(
    [ift_cache[f"blocks.{layer}.attn.hook_result"] for layer in range(24)],
    dim=0,
)[:, 0, -1, :, :]
gpt2_head_outs = torch.stack(
    [gpt2_cache[f"blocks.{layer}.attn.hook_result"] for layer in range(24)],
    dim=0,
)[:, 0, -1, :, :]
print(ift_head_outs.shape)

print(
    F.cosine_similarity(
        ift_head_outs[layer, :, :],
        next_tok_vec.to(ift_head_outs.device).unsqueeze(0),
        dim=1,
    )
)
breakpoint()
print(gpt2_cache.keys())
print(
    F.cosine_similarity(
        gpt2_head_outs[layer, :, :], next_tok_vec.unsqueeze(0), dim=1
    )
)

breakpoint()

print(
    einsum(
        "head d_model, d_model -> head",
        ift_head_outs[layer],
        next_tok_vec.to(ift_head_outs.device),
    )
)
print(
    einsum(
        "head d_model, d_model -> head", gpt2_head_outs[layer], next_tok_vec
    )
)

# %%

print(gpt2_cache[f"blocks.{layer}.attn.hook_attn_scores"].shape)
print(gpt2_cache[f"blocks.{layer}.attn.hook_pattern"].shape)
patterns = gpt2_cache[f"blocks.{layer}.attn.hook_pattern"].squeeze()
for head_idx in range(patterns.shape[0]):
    print(patterns[head_idx, -1, :])


breakpoint()

# %%

# 1024
mlp_17 = gpt2_cache["blocks.17.hook_resid_mid"][:, -1, :].clone()
print(mlp_17.shape)
mlp_17_proj_vocab = einsum(
    "batch d_model, d_model vocab -> batch vocab", mlp_17, unembed
)
print(tokenizer.batch_decode(mlp_17_proj_vocab.topk(k=10).indices[0]))


mlp_mid = gpt2_cache[f"blocks.17.mlp.hook_post"][0, -1, :]
high_acts = mlp_mid.topk(k=50)
print(high_acts)

# [k, d_model]
high_w_out = gpt2.blocks[17].mlp.W_out[high_acts.indices[:50]]
proj_vocab = einsum("k d_model, d_model vocab -> k vocab", high_w_out, unembed)
for idx in range(high_w_out.shape[0]):
    print(tokenizer.batch_decode(proj_vocab.topk(k=30).indices[idx]))


# %%


cos_scores = F.cosine_similarity(
    gpt2.blocks[17].mlp.W_out, next_tok_vec.unsqueeze(dim=0), dim=1
)
top_cos = cos_scores.topk(k=20)
print(top_cos)
high_w_out = gpt2.blocks[17].mlp.W_out[top_cos.indices]
proj_vocab = einsum("k d_model, d_model vocab -> k vocab", high_w_out, unembed)
for idx in range(high_w_out.shape[0]):
    print(tokenizer.batch_decode(proj_vocab.topk(k=30).indices[idx]))


# %%

decomps_lbsd, labels = gpt2_cache.decompose_resid(return_labels=True)

print(decomps.shape)
mlp17_idx = labels.index("17_mlp_out")
print(labels)
print(mlp17_idx)
_decomp = decomps_lbsd[mlp17_idx, 0, -1, :]

top_cos = F.cosine_similarity(
    _decomp.unsqueeze(0), gpt2.blocks[17].mlp.W_out, dim=1
).topk(k=20)
print(top_cos)
proj_vocab = einsum(
    "k d_model, d_model vocab -> k vocab",
    gpt2.blocks[17].mlp.W_out[top_cos.indices],
    unembed,
)
print(proj_vocab)
for layer in range(proj_vocab.shape[0]):
    print(tokenizer.batch_decode(proj_vocab[layer].topk(k=20).indices))

# proj_vocab = einsum("layer d_model, d_model vocab -> layer vocab", _decomp, unembed)
# for layer in range(decomps_lbsd.shape[0]):
#    print(tokenizer.batch_decode(proj_vocab[layer].topk(k=30).indices))


# %%


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
    sns.lineplot(data=plot_data, x="Layer", y=y, hue="Model", ax=ax)

# %%
