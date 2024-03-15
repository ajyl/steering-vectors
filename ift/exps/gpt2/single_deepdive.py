"""
Deep dive on single samples
"""

import os
import json
from tqdm import tqdm
from fancy_einsum import einsum
import torch
from ift.interp_utils import get_model_and_tokenizer, load_steer_vecs
from steering_vectors.steering_vector import hooked_generate
from steering_vectors.record_activations import record_activations
from steering_vectors.torch_utils import get_module
from ift.data_utils import (
    get_batch,
    tokenize_data,
    load_cached_data,
    build_ift_prompts,
)


#def build_activation_stack(recorded_activations):
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

    model, tokenizer = get_model_and_tokenizer(
        config["ift_model_path"], config["tokenizer"], device_map="auto"
    )

    print("Generating for GPT2 + steer...")
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

    with record_activations(
        model,
        layer_types=[
            "decoder_block",
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
        ],
    ) as recorded_activations:
        # output = model.generate(
        #    prompt_input_ids.to(model.device),
        #    attention_mask=attention_mask.to(model.device),
        #    do_sample=False,
        #    max_new_tokens=max_new_tokens,
        #    pad_token_id=tokenizer.pad_token_id,
        #    # use_cache=False,
        # )
        logits = model.forward(
            prompt_input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )

    breakpoint()
    acts_ld = torch.stack(
        [
            _acts[0][0, -1, :].to("cuda:0")
            for _acts in recorded_activations.values()
        ],
        dim=0,
    )
    breakpoint()
    # output_text = tokenizer.batch_decode(
    #    output[:, prompt_shape[1] :], skip_special_tokens=True
    # )

    # prediction = output[:, prompt_shape[1]]
    logits_bsv = logits.logits
    prediction = logits_bsv[:, -1, :].argmax(-1)

    # "1": 16
    # "I": 40
    # IFT: ['1', '2', 'Ingredients', '6', '3', '*', '##', 'For', '4', '``']
    # IFT: [   16,    17, 41222,    21,    18,     9,  2235,  1890,    19, 15506]

    # GPT2: ['I', '1', '[', 'Ingredients', 'The', 'This', 'You', 'A', '*', 'For']
    # GPT2: [   40,    16,    58, 41222,   464,  1212,  1639,    32,     9,  1890]
    topk_preds = logits_bsv[:, -1, :].topk(k=10).indices
    topk_preds_toks = tokenizer.batch_decode(topk_preds)

    unembed_vd = model.lm_head.weight
    acts_ld = torch.stack(
        [
            _acts[0][0, -1, :].to("cuda:0")
            for _acts in recorded_activations.values()
        ],
        dim=0,
    )
    logit_lens = einsum(
        "layer d_model, d_model vocab -> layer vocab",
        acts_ld,
        unembed_vd.transpose(0, 1),
    )
    breakpoint()

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

    model, tokenizer = get_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="auto"
    )
    model, tokenizer = get_hooked_model_and_tokenizer(
        config["model"], config["tokenizer"], device_map="auto"
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

    with record_activations(
        model, layer_type="decoder_block"
    ) as recorded_activations:
        # output = model.generate(
        #    prompt_input_ids.to(model.device),
        #    attention_mask=attention_mask.to(model.device),
        #    do_sample=False,
        #    max_new_tokens=max_new_tokens,
        #    pad_token_id=tokenizer.pad_token_id,
        # )
        logits = model.forward(
            prompt_input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )

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
