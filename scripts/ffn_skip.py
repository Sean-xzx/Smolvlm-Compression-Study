"""Instance-only Llama FFN branch skip, adapted from transformers v4.57.6.

Source: transformers/models/llama/modeling_llama.py LlamaDecoderLayer.forward
(Apache-2.0). Attention half is unchanged; norm2 and MLP are omitted.
Original full source and patch are saved in the evening evidence directory.
"""
import hashlib
import inspect
from types import MethodType

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils.deprecation import deprecate_kwarg


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def skip_forward(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=False, cache_position=None,
                 position_embeddings=None, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        use_cache=use_cache, cache_position=cache_position,
        position_embeddings=position_embeddings, **kwargs,
    )
    hidden_states = residual + hidden_states
    # Entire norm2 + MLP branch is absent. Original parameters remain registered.
    return hidden_states


class SkipPatch:
    def __init__(self, model, layers_1based, expected_source_sha256):
        layers = model.model.text_model.layers
        selected = list(layers_1based)
        if any(type(i) is not int or not 1 <= i <= len(layers) for i in selected):
            raise ValueError("Skip layers must be integer one-based Llama layer numbers")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate Skip layer number")
        source = inspect.getsourcefile(LlamaDecoderLayer)
        with open(source, "rb") as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != expected_source_sha256:
            raise ValueError("Installed Llama source differs from the reviewed version")
        if any(type(layer) is not LlamaDecoderLayer for layer in layers):
            raise TypeError("Only the reviewed LlamaDecoderLayer is supported")
        if any(hasattr(layer, "_ffn_skip_original_forward") for layer in layers):
            raise ValueError("A Skip patch is already installed")
        self.layers_1based, self.originals = selected, []
        for number in selected:
            layer = layers[number - 1]
            self.originals.append((layer, layer.forward))
            layer._ffn_skip_original_forward = layer.forward
            layer.forward = MethodType(skip_forward, layer)

    def restore(self):
        for layer, original in self.originals:
            layer.forward = original
            del layer._ffn_skip_original_forward
        self.originals.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
