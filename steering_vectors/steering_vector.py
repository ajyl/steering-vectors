from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Generator, overload

from tqdm import tqdm
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from .torch_utils import get_module, untuple_tensor

PatchDeltaOperator = Callable[[Tensor, Tensor], Tensor]


@dataclass
class SteeringPatchHandle:
    """
    A handle that can be used to remove a steering patch from a model after
    running `steering_vector.patch_activations()`.
    """

    model_hooks: list[RemovableHandle]

    def remove(self) -> None:
        """Remove the steering patch from the model"""
        for hook in self.model_hooks:
            hook.remove()


@dataclass
class SteeringVector:
    """A steering vector that can be applied to a model."""

    layer_activations: dict[int, Tensor]
    layer_type: LayerType = "decoder_block"

    def patch_activations(
        self,
        model: nn.Module,
        layer_config: ModelLayerConfig | None = None,
        operator: PatchDeltaOperator | None = None,
        multiplier: float = 1.0,
        min_token_index: int | None = None,
        token_indices: list[int] | slice | Tensor | None = None,
    ) -> SteeringPatchHandle:
        """
        Patch the activations of the given model with this steering vector.
        This will modify the model in-place, and return a handle that can be used to undo the patching.
        This method does the same thing as `apply`, but requires manually undoing the patching to
        restore the model to its original state. For most cases, `apply` is easier to use. Tokens to patch
        can be selected using either `min_token_index` or `token_indices`, but not both. If neither is provided,
        all tokens will be patched.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the steering vector
                and returns a modified vector that is added to the original activation.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
            min_token_index: The minimum token index to apply the patch to. Default is None.
            token_indices: Either a list of token indices to apply the patch to, a slice, or a mask tensor. Default is None.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> handle = steering_vector.patch_activations(model)
            >>> model.forward(...)
            >>> handle.remove()
        """
        assert (min_token_index is None) or (
            token_indices is None
        ), "Can not pass both min_token_index and token_indices"
        if isinstance(token_indices, Tensor):
            assert torch.all(
                torch.logical_or(token_indices == 0, token_indices == 1)
            ), "token_indices tensor must be a mask (containing only 0s and 1s)"
        token_indices = (
            token_indices
            if token_indices is not None
            else slice(min_token_index, None)
        )

        layer_config = guess_and_enhance_layer_config(
            model, layer_config, self.layer_type
        )
        hooks: list[RemovableHandle] = []
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not provided in layer config"
            )
        matcher = layer_config[self.layer_type]
        matching_layers = collect_matching_layers(model, matcher)
        for layer_num, target_activation in self.layer_activations.items():
            layer_name = matching_layers[layer_num]

            target_activation = multiplier * self.layer_activations[layer_num]

            module = get_module(model, layer_name)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_additive_hook(
                    target_activation.reshape(1, 1, -1),
                    token_indices,
                    operator,
                )
            )
            hooks.append(handle)
        return SteeringPatchHandle(hooks)

    @contextmanager
    def apply(
        self,
        model: nn.Module,
        layer_config: ModelLayerConfig | None = None,
        operator: PatchDeltaOperator | None = None,
        multiplier: float = 1.0,
        min_token_index: int = 0,
        token_indices: list[int] | slice | Tensor | None = None,
        num_timesteps: int = 0,
    ) -> Generator[None, None, None]:
        """
        Apply this steering vector to the given model. Tokens to patch
        can be selected using either `min_token_index` or `token_indices`, but not both.
        If neither is provided, all tokens will be patched.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the steering vector
                and returns a modified vector that is added to the original activation.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
            min_token_index: The minimum token index to apply the patch to. Default is None.
            token_indices: Either a list of token indices to apply the patch to, a slice, or a mask tensor. Default is None.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> with steering_vector.apply(model):
            >>>     model.forward(...)
        """
        try:
            handle = self.patch_activations(
                model=model,
                layer_config=layer_config,
                operator=operator,
                multiplier=multiplier,
                min_token_index=min_token_index,
                token_indices=token_indices,
            )
            yield
        finally:
            handle.remove()

    # types copied from torch.Tensor

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> "SteeringVector":
        ...

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> "SteeringVector":
        ...

    @overload
    def to(
        self, other: Tensor, non_blocking: bool = False, copy: bool = False
    ) -> "SteeringVector":
        ...

    def to(self, *args: Any, **kwargs: Any) -> "SteeringVector":
        """
        Return a new steering vector moved to the given device/dtype.

        This method calls ``torch.Tensor.to`` on each of the layer activations.
        """
        layer_activations = {
            layer_num: act.to(*args, **kwargs)
            for layer_num, act in self.layer_activations.items()
        }
        return replace(self, layer_activations=layer_activations)

    def dump(self, filepath: str) -> None:
        """
        Save layer_activations.
        """
        torch.save(
            {
                "layer_activations": self.layer_activations,
                "layer_type": self.layer_type,
            },
            filepath,
        )

    @classmethod
    def load_from_file(cls, filepath: str) -> None:
        """
        Initialize layer_activations, layer_type from cached file.
        """
        cached = torch.load(filepath)
        return cls(
            layer_activations=cached["layer_activations"],
            layer_type=cached["layer_type"],
        )


def _create_additive_hook(
    target_activation: Tensor,
    token_indices: list[int] | slice | Tensor,
    operator: PatchDeltaOperator | None = None,
) -> Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        original_tensor = untuple_tensor(outputs)
        act = target_activation.to(original_tensor.device)
        delta = act
        if operator is not None:
            delta = operator(original_tensor, act)
        if isinstance(token_indices, Tensor):
            mask = token_indices
        else:
            mask = torch.zeros(original_tensor.shape[1])
            mask[token_indices] = 1
        mask = mask.reshape(1, -1, 1)
        mask = mask.to(original_tensor.device)
        original_tensor[None] = original_tensor + (mask * delta)
        return outputs

    return hook_fn


def hooked_generate(
    model,
    tokenizer,
    steering_vectors,
    input_ids,
    gen_timesteps,
    multiplier,
    **model_kwargs,
):
    """
    Hmm.
    """

    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id]).to(
        input_ids.device
    )
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    ngram_processor = NoRepeatNGramLogitsProcessor(3)
    for timestep in range(gen_timesteps):
        handle = None
        # if timestep == 0:
        #    steering_vector = steering_vectors[0]
        # else:
        #    steering_vector = steering_vectors[2]
        steering_vector = steering_vectors.get(timestep)

        # if steering_vector is None:
        #    steering_vectors.get(19)

        if steering_vector is not None:

            # if timestep == 0:
            #    steering_vector.layer_activations.pop(23)

            handle = steering_vector.patch_activations(
                model, multiplier=multiplier.get(timestep, 0)
            )

        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            **model_kwargs,
        )

        model_inputs["use_cache"] = False

        with torch.inference_mode():
            outputs = model.forward(
                **model_inputs,
                # use_cache=False,
            )

        next_token_logits = outputs.logits[:, -1, :].clone()

        # next_tokens_scores = ngram_processor(input_ids, next_token_logits)

        if timestep == 0:
            # " The"
            next_token_logits[:, 383] = -1e10
        #    # " Here"
        #    #next_token_logits[:, 3423] = -1e10

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
        if handle is not None:
            handle.remove()

        if unfinished_sequences.max() == 0:
            break

    return input_ids
