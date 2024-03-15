"""
Module Doc String
"""

import os

import torch
import torch.nn.functional as F
import pandas as pd
import einops
import seaborn as sns
from fancy_einsum import einsum

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs


def check_steer_vec_direction_vocab(steer_vecs, config):
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )
    unembed_vd = model.lm_head.weight
    k = 30

    for timestep in steer_vecs.keys():

        steer_vec_ld = torch.stack(
            [
                steer_vecs[timestep].layer_activations[layer]
                for layer in range(24)
            ],
            dim=0,
        )

        dot_prods_lv = einsum(
            "layer d_model, d_model vocab -> layer vocab",
            steer_vec_ld.to(unembed_vd.device),
            unembed_vd.transpose(0, 1),
        )
        top_vocabs_lk = dot_prods_lv.topk(k=k, dim=1).indices

        for layer in range(24):
            print(f"Layer {layer}")
            top_tokens = tokenizer.batch_decode(top_vocabs_lk[layer])
            print(top_tokens)

        dot_prods_lv = einsum(
            "layer d_model, d_model vocab -> layer vocab",
            model.transformer.ln_f(steer_vec_ld).to(unembed_vd.device),
            unembed_vd.transpose(0, 1),
        )
        top_vocabs_lk = dot_prods_lv.topk(k=k, dim=1).indices
        top_tokens2 = tokenizer.batch_decode(top_vocabs_lk)

        breakpoint()
        print("z")


def steer_vec_cos_sim_per_timestep(steer_vecs, config):
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )

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

    fig = plt.figure(figsize=(12, 12))
    num_cols = 5
    gs = GridSpec(4, num_cols)
    for timestep in range(steer_vecs_tld.shape[0]):
        curr_row = timestep // num_cols
        curr_col = timestep % num_cols

        steer_vec_ld = steer_vecs_tld[timestep]
        cos_norm = steer_vec_ld / steer_vec_ld.norm(dim=1)[:, None]
        cos_scores = torch.mm(cos_norm, cos_norm.transpose(0, 1))

        ax = fig.add_subplot(gs[curr_row, curr_col])
        sns.heatmap(
            cos_scores,
            ax=ax,
            cmap="magma_r",
            vmax=1.0,
            vmin=0,
            # annot=True,
            # fmt=".2f",
        )
        ax.set_title(f"Timestep {timestep}")
    fig.suptitle(f"Cos-sim of steer_vec across layers per timestep.")
    fig.savefig("heatmap_at_timestep.png")


def steer_vec_cos_sim_across_timesteps(steer_vecs, config):
    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"]
    )

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

    fig = plt.figure(figsize=(12, 12))
    num_cols = 6
    gs = GridSpec(4, num_cols)

    # across timesteps.
    for layer in range(24):
        curr_row = layer // num_cols
        curr_col = layer % num_cols

        steer_vec_td = steer_vecs_tld[:, layer, :]
        cos_norm = steer_vec_td / steer_vec_td.norm(dim=1)[:, None]
        cos_scores = torch.mm(cos_norm, cos_norm.transpose(0, 1))

        ax = fig.add_subplot(gs[curr_row, curr_col])
        sns.heatmap(cos_scores, ax=ax, cmap="magma_r", vmax=1.0, vmin=0)

        ax.set_title(f"Layer {layer}")
    fig.suptitle(f"Cos-sim of steer_vec across timesteps per layer.")
    fig.savefig("heatmap_at_layer.png")


def unembed_shift_vs_component_shift(config, component):
    """
    Check shift in weights.
    """
    assert component in ["mlp.w_in", "mlp.w_out", "attn_v", "attn_o"]
    gpt2, _ = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map=0
    )
    ift = torch.load(config["ift_state_dict"])["state"]
    n_heads = gpt2.config.n_head

    unembed_shift_vd = (
        ift["lm_head.weight"].to(gpt2.device) - gpt2.lm_head.weight
    )
    diff_norms = unembed_shift_vd.norm(dim=1)
    top_diffs = torch.abs(diff_norms).topk(k=10000)

    top_diff_directions_kd = unembed_shift_vd[top_diffs.indices]

    mean_shift_d = top_diff_directions_kd.mean(dim=0)

    layer_cos_sims = []
    for name, param in gpt2.named_parameters():
        layer = None
        if name.startswith("transformer.h"):
            layer = int(name.split(".")[2])

        ift_param = ift[name].to(param.device)

        if component == "mlp.w_in":
            if ".mlp.c_fc.weight" not in name:
                continue

            diff = ift_param - param
            # [d_model, d_mlp]
            cos_sim = F.cosine_similarity(
                gpt2.transformer.ln_f(diff.transpose(0, 1)).T,
                mean_shift_d.unsqueeze(-1),
                dim=0,
            )
            layer_cos_sims.append(cos_sim)

        elif component == "mlp.w_out":
            if ".mlp.c_proj.weight" not in name:
                continue

            diff = ift_param - param
            # [d_mlp, d_model]
            cos_sim = F.cosine_similarity(
                gpt2.transformer.ln_f(diff), mean_shift_d.unsqueeze(0), dim=1
            )
            layer_cos_sims.append(cos_sim)

        elif component == "attn_v":
            if ".attn.c_attn.weight" not in name:
                continue

            gpt2_q, gpt2_k, gpt2_v = torch.tensor_split(diff, 3, dim=1)
            gpt2_v = einops.rearrange(gpt2_v, "m (i h)->i h m", i=n_heads)
            cos_sim = F.cosine_similarity(
                gpt2_v, mean_shift_d.unsqueeze(0).unsqueeze(0), dim=2
            )
            layer_cos_sims.append(cos_sim)

        elif component == "attn_o":
            if ".attn.c_proj.weight" not in name:
                continue

            gpt2_o = einops.rearrange(diff, "(i h) m ->i h m", i=n_heads)
            cos_sim = F.cosine_similarity(
                gpt2_o, mean_shift_d.unsqueeze(0).unsqueeze(0), dim=2
            )
            layer_cos_sims.append(cos_sim)

    raw_cosine_data = []
    for layer in range(24):
        for _idx in range(layer_cos_sims[layer].shape[0]):
            raw_cosine_data.append(
                {
                    "layer": layer,
                    "cos_sim": round(layer_cos_sims[layer][_idx].item(), 2),
                }
            )
    cos_sim_df = pd.DataFrame(raw_cosine_data)

    num_cols = 6

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, num_cols)

    for layer in range(24):
        curr_row = layer // num_cols
        curr_col = layer % num_cols

        ax = fig.add_subplot(gs[curr_row, curr_col])

        sns.histplot(
            data=cos_sim_df[cos_sim_df.layer == layer],
            x="cos_sim",
            ax=ax,
            stat="probability",
            alpha=1,
            element="poly",
        )
        ax.set(xticks=[-1, -0.5, 0, 0.5, 1])
        ax.set(yticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])

    fig.savefig(f"components_vs_embeddings_shifts_{component}.png")


def steer_vec_vs_component_shift(steer_vecs, config, component):
    """
    Check shift in weights.
    """
    assert component in ["mlp.w_in", "mlp.w_out", "attn_v", "attn_o"]
    gpt2, _ = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map=0
    )
    n_heads = gpt2.config.n_head
    gpt2 = gpt2.state_dict()
    ift = torch.load(config["ift_state_dict"])["state"]

    layer_cos_sims = []

    for layer in range(24):

        steer_vec = steer_vecs[0].layer_activations[layer]

        if component == "mlp.w_in":
            ift_param = ift[f"transformer.h.{layer}.mlp.c_fc.weight"].to(
                "cuda:0"
            )
            gpt2_param = gpt2[f"transformer.h.{layer}.mlp.c_fc.weight"].to(
                "cuda:0"
            )

            diff = ift_param - gpt2_param
            # [d_model, d_mlp]
            cos_sim = F.cosine_similarity(
                diff.to(steer_vec.device),
                steer_vec.unsqueeze(-1),
                dim=0,
            )
            layer_cos_sims.append(cos_sim)

        elif component == "mlp.w_out":

            if layer == 0:
                continue

            ift_param = ift[f"transformer.h.{layer-1}.mlp.c_proj.weight"].to(
                "cuda:0"
            )
            gpt2_param = gpt2[f"transformer.h.{layer-1}.mlp.c_proj.weight"].to(
                "cuda:0"
            )

            diff = ift_param - gpt2_param
            # [d_mlp, d_model]
            cos_sim = F.cosine_similarity(
                diff.to(steer_vec.device), steer_vec.unsqueeze(0), dim=1
            )
            layer_cos_sims.append(cos_sim)

        elif component == "attn_v":
            ift_param = ift[f"transformer.h.{layer}.attn.c_attn.weight"].to(
                "cuda:0"
            )
            gpt2_param = gpt2[f"transformer.h.{layer}.attn.c_attn.weight"].to(
                "cuda:0"
            )
            _, _, ift_v = torch.tensor_split(ift_param, 3, dim=1)
            _, _, gpt2_v = torch.tensor_split(gpt2_param, 3, dim=1)

            diff = ift_v - gpt2_v
            # diff = einops.rearrange(gpt2_v, "m (i h)->i h m", i=n_heads)
            cos_sim = F.cosine_similarity(
                diff.to(steer_vec.device), steer_vec.unsqueeze(1), dim=0
            )
            layer_cos_sims.append(cos_sim)

        elif component == "attn_o":
            if layer == 0:
                continue
            ift_param = ift[f"transformer.h.{layer-1}.attn.c_proj.weight"].to(
                "cuda:0"
            )
            gpt2_param = gpt2[
                f"transformer.h.{layer-1}.attn.c_proj.weight"
            ].to("cuda:0")
            diff = ift_param - gpt2_param
            cos_sim = F.cosine_similarity(
                diff.to(steer_vec.device), steer_vec.unsqueeze(0), dim=1
            )
            layer_cos_sims.append(cos_sim)

    raw_cosine_data = []
    num_layers = 24
    if component in ["mlp.w_out", "attn_o"]:
        num_layers = 23
    for layer in range(num_layers):
        for _idx in range(layer_cos_sims[layer].shape[0]):
            raw_cosine_data.append(
                {
                    "layer": layer,
                    "cos_sim": round(layer_cos_sims[layer][_idx].item(), 2),
                }
            )
    cos_sim_df = pd.DataFrame(raw_cosine_data)

    num_cols = 6

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, num_cols)

    for layer in range(24):
        curr_row = layer // num_cols
        curr_col = layer % num_cols

        ax = fig.add_subplot(gs[curr_row, curr_col])

        sns.histplot(
            data=cos_sim_df[cos_sim_df.layer == layer],
            x="cos_sim",
            ax=ax,
            stat="probability",
            alpha=1,
            element="poly",
        )
        ax.set(xticks=[-1, -0.5, 0, 0.5, 1])
        ax.set(yticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])

    fig.savefig(f"components_vs_steer_vecs_{component}.png")

def svd(steer_vecs, config):
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
    svd = torch.linalg.svd(_steer_vecs.tranpose(0, 1))

    svd_U = svd.U # [1024, 1024]
    return svd_U



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
        "ckpt_dir": "/home/repos/steering-vectors/ckpts/",
    }
    steer_vecs = load_steer_vecs(
        os.path.join(config["ckpt_dir"], "steer_vec_series_longer.pt")
    )
    #check_steer_vec_direction_vocab(steer_vecs, config)
    #steer_vec_cos_sim_per_timestep(steer_vecs, config)
    #steer_vec_cos_sim_across_timesteps(steer_vecs, config)

    # unembed_shift_vs_component_shift(config, "mlp.w_out")
    # steer_vec_vs_component_shift(steer_vecs, config, "mlp.w_in")
    #steer_vec_vs_component_shift(steer_vecs, config, "attn_o")
    #steer_vec_vs_component_shift(steer_vecs, config, "attn_v")

    svd(steer_vecs, config)


if __name__ == "__main__":
    main()
