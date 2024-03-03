"""
Utility functions for data.
"""

import json
from transformers import AutoTokenizer


def get_batch(data, idx, batch_size):
    """
    Get single batch.
    """
    return {
        "prompts": data["prompts"][idx : idx + batch_size],
        "prompt_input_ids": data["prompt_input_ids"][idx : idx + batch_size],
        "prompt_attention_mask": data["prompt_attention_mask"][
            idx : idx + batch_size
        ],
    }


def tokenize_data(data, config):
    """
    Tokenize data.

    return:
    {
        "titles": List[str],
        "questions": List[str],
        "responses": List[str],
        "prompts": List[str],
        "continuations": List[str],
        "prompt_input_ids": Torch.LongTensor,
        "prompt_attention_mask": Torch.LongTensor,
        "continuation_input_ids": Torch.LongTensor,
        "continuation_attention_mask": Torch.LongTensor,
    }
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if config["model"].startswith("gpt2"):
        tokenizer.padding_side = "left"

    prompts_text = [x["prompt"] for x in data]
    prompts = tokenizer(
        prompts_text,
        max_length=config["max_prompt_length"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "prompts": prompts_text,
        "prompt_input_ids": prompts["input_ids"],
        "prompt_attention_mask": prompts["attention_mask"],
    }


def load_cached_data(filepath):
    """
    Load cached data.
    """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)
    return data


def build_ift_prompts(data, config):
    """
    Add "prompt", "prompt_ift" fields.
    """
    formatted = []
    for sample in data:

        sample["prompt"] = " ".join(
            [
                config["prompt_token"],
                sample["question"],
                config["response_token"],
            ]
        )
        formatted.append(sample)

    return formatted
