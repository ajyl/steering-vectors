"""
Experiment with patching some timesteps.
"""

import os
import json
from tqdm import tqdm
import torch
from torch import nn
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs
from steering_vectors.record_activations import record_activations
from ift.exps.llama2.data_utils import get_batch, load_lima_data, tokenize_data


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def stitch_lm_head(model, config):
    """
    Return Llama with lm head swapped out.
    """
    model_state_dict = model.state_dict()
    model_state_dict["model.norm.weight"] = torch.load(
        "pt_files/model.norm.weight.pt"
    ).to(model.device)
    model_state_dict["lm_head.weight"] = torch.load("pt_files/lm_head.weight.pt").to(
        model.device
    )

    model.load_state_dict(model_state_dict)
    return model



@torch.no_grad()
def patch_generate(model, tokenizer, hidden_states, batch, config, **model_kwargs):
    """
    Patch model.
    """
    max_new_tokens = config["max_new_tokens"]

    def _swap_hidden_states(_timestep, prompt_shape, _hidden_state):
        # _hidden_state: [batch, seq, d_model]

        def _hook_fn_post(_module, _inputs, _outputs):

            # [batch, seq, d_model]
            original_tensor = untuple_tensor(_outputs)
            # original_tensor[:, _timestep, :] = _hidden_state[:, _timestep, :].to(
            # original_tensor[:, 0, :] = _hidden_state[:, 0, :].to(
            #    original_tensor.device
            # )
            # original_tensor[:, prompt_shape + _timestep - 1, :] = _hidden_state[
            #   :, prompt_shape + _timestep - 1, :
            # ].to(original_tensor.device)
            original_tensor[:, -1, :] = _hidden_state[
                :, prompt_shape + _timestep - 1, :
            ].to(original_tensor.device)
            return _outputs

        def _hook_fn_pre(_module, _inputs):
            # [batch, seq, d_model]
            original_input = untuple_tensor(_inputs)
            original_input[:, _timestep, :] = _hidden_state[:, _timestep, :].to(
                original_input.device
            )
            original_input[:, prompt_shape + _timestep - 1, :] = _hidden_state[
                :, prompt_shape + _timestep - 1, :
            ].to(original_input.device)
            return _inputs

        return _hook_fn_post

    input_ids = batch["prompt_input_ids"].to(model.device)
    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id]).to(model.device)
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    start_timestep = 0
    end_timestep = 16
    prompt_shape = input_ids.shape[1]
    for timestep in tqdm(range(start_timestep, end_timestep)):

        hooks = []
        if timestep < 5:
            for layer_name, acts in hidden_states.items():
                # acts: [batch, seq (prompt_size + max_new_tokens), d_model]
                if layer_name != "model.layers.31":
                    continue

                module = get_module(model, layer_name)
                # handle = module.register_forward_pre_hook(
                #    _swap_hidden_states(timestep, prompt_shape, acts)
                # )
                handle = module.register_forward_hook(
                    _swap_hidden_states(timestep, prompt_shape, acts)
                )
                hooks.append(handle)

        attn_mask = (input_ids != tokenizer.pad_token_id).to(model.device)
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attn_mask,
            **model_kwargs,
        )

        model_inputs["use_cache"] = False

        with torch.inference_mode():
            outputs = model.forward(
                **model_inputs,
            )

        next_token_logits = outputs.logits[:, -1, :].clone()

        # if timestep == 0:
        #    # " The"
        #    next_token_logits[:, 383] = -1e10
        #    # " Here"
        #    #next_token_logits[:, 3423] = -1e10
        # if timestep == 2:
        #    # 'Comment'
        #    next_token_logits[:, 20001] = -1e10

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1)
            .ne(eos_token_id_tensor.unsqueeze(1))
            .prod(dim=0)
        )
        for handle in hooks:
            handle.remove()

        if unfinished_sequences.max() == 0:
            break

    output_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    for _tmp in output_text:
        print(_tmp)
    breakpoint()
    return input_ids


@torch.no_grad()
def get_hidden_state(model, tokenizer, batch, config):
    """
    Get hidden state from forward pass on a batch.
    """
    max_new_tokens = config["max_new_tokens"]
    attn_mask = batch["prompt_attention_mask"]
    position_ids = attn_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn_mask == 0, 1)

    with record_activations(model, layer_types=["decoder_block"]) as record:
        output = model.generate(
            batch["prompt_input_ids"].to(model.device),
            attention_mask=batch["prompt_attention_mask"].to(model.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    hidden_states = {}
    for layer_name, acts in record.items():
        hidden_states[layer_name] = torch.cat(acts, dim=1).cpu()

    # [batch, layers, seq, d_model]
    # hidden_states = torch.stack(_hidden_states.value(), dim=1)
    # return hidden_states
    text_out = tokenizer.batch_decode(output, skip_special_tokens=True)
    print("Original:")
    for _tmp in text_out:
        print("----------------")
        print(_tmp)

        print("----------------")

    return hidden_states


def patch(data, config):
    """
    Patch experiment!
    """
    patch_config = config["patch"]

    llama, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="cuda:0"
    )
    ift, _ = get_model_and_tokenizer(
        config["ift_model_path"], config["tokenizer"], device_map="cuda:1"
    )

    batch_size = config["batch_size"]
    # num_samples = data.shape[0]
    num_samples = 6

    stitched = stitch_lm_head(llama, config)

    for idx in tqdm(range(0, num_samples, batch_size)):
        batch = get_batch(data, idx, batch_size)
        ift_hidden_states = get_hidden_state(ift, tokenizer, batch, config)

        
        patch_generate(stitched, tokenizer, ift_hidden_states, batch, patch_config)


def main():
    """Driver"""
    config = {
        "model": "meta-llama/Llama-2-7b-hf",
        # "model": "/nfs/turbo/coe-mihalcea/ajyl/llama2",
        "tokenizer": "meta-llama/Llama-2-7b-hf",
        "prompt_token": "[Instruction]",
        "response_token": "[Response]",
        "batch_size": 4,
        "max_prompt_length": 512,
        "max_response_length": 128,
        "max_new_tokens": 10,
        "cache_dir": "/home/ajyl/steering-vectors/data/cache/no_robots",
        "num_generate": 16,
        "ift_state_dict": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/LATEST/ift_model.pt",
        # "ift_model_path": "/scratch/mihalcea_root/mihalcea0/ajyl/ift/ajyl/llama_v3/llama_ift_LATEST",
        "ift_model_path": "/nfs/turbo/coe-mihalcea/ajyl/llama_ift_LATEST",
        "ckpt_dir": "/home/ajyl/steering-vectors/ckpts/llama2",
        "patch": {
            "max_new_tokens": 10,
        },
    }
    print("Loading lima data.")
    data = load_lima_data("train", config)
    data = tokenize_data(data, config)

    patch(data, config)


if __name__ == "__main__":
    main()
