"""Instance-level sparse Idefics3 Encoder + postnorm zero-grid adapter.

Encoder loop and selected layer arithmetic adapt transformers v4.57.6
modeling_idefics3.py (Apache-2.0). Native Attention, FFN, postnorm, Connector
and LLM remain modules from that implementation. Selection follows the frozen
local contract; no AudioMAE code/repository is claimed as executed or copied.
"""
import hashlib
import inspect
import math
from types import MethodType

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.idefics3.modeling_idefics3 import Idefics3EncoderLayer


def attention_scores(weights, valid):
    """Incoming attention per key: mean over heads and valid query rows."""
    if weights is None or weights.ndim != 4 or weights.shape[-2:] != (valid.shape[1], valid.shape[1]):
        raise ValueError("Need native eager [blocks, heads, query, key] attention")
    n = valid.sum(1)
    if (n == 0).any():
        raise ValueError("An image block has no valid patch tokens")
    # Float32 reduction; never average over key instead of query.
    per_query = weights.float().mean(1)
    score = (per_query * valid[:, :, None]).sum(1) / n[:, None]
    if not torch.isfinite(score[valid]).all():
        raise ValueError("Nonfinite attention score")
    return score.masked_fill(~valid, -torch.inf)


def select_tokens(hidden, positions, valid, scores, rate):
    """Stable Top-K set, original order, physical padding to max effective K."""
    if not 0 < rate < 1:
        raise ValueError("Selection is only for strict pruning; rate 1 must bypass")
    counts = valid.sum(1).tolist()
    if any(n == 0 for n in counts):
        raise ValueError("Cannot prune an empty valid block")
    ks = [max(1, math.ceil(rate * n)) for n in counts]
    bsz, _, width = hidden.shape
    physical = max(ks)
    output = hidden.new_zeros(bsz, physical, width)
    out_pos = positions.new_full((bsz, physical), -1)
    out_valid = valid.new_zeros(bsz, physical)
    selections = []
    for b, k in enumerate(ks):
        candidates = torch.nonzero(valid[b], as_tuple=False).flatten()
        # Establish the explicit tie order, even if current positions were reordered.
        candidates = candidates[torch.argsort(positions[b, candidates], stable=True)]
        ranking = torch.argsort(scores[b, candidates], descending=True, stable=True)
        chosen = candidates[ranking[:k]]
        chosen = chosen[torch.argsort(positions[b, chosen], stable=True)]
        output[b, :k] = hidden[b, chosen]
        out_pos[b, :k] = positions[b, chosen]
        out_valid[b, :k] = True
        selections.append(chosen.tolist())
    return output, out_pos, out_valid, selections, ks


def restore_grid(features, positions, valid, original_length):
    """Only postnorm retained features are scattered; padding has no grid slot."""
    grid = features.new_zeros(features.shape[0], original_length, features.shape[-1])
    occupied = valid.new_zeros(features.shape[0], original_length)
    for b in range(features.shape[0]):
        indices = positions[b, valid[b]]
        if (indices < 0).any() or (indices >= original_length).any() or indices.unique().numel() != indices.numel():
            raise ValueError("Invalid or duplicated original grid position")
        grid[b, indices] = features[b, valid[b]]
        occupied[b, indices] = True
    return grid, occupied


class EncoderTopKPatch:
    def __init__(self, model, config, contract, trace=True):
        self.model, self.trace = model, trace
        self.encoder, self.connector = model.model.vision_model.encoder, model.model.connector
        self.layers = self.encoder.layers
        selected, rates = config["layers_1based"], config["keep_rates"]
        if len(selected) != len(rates) or len(set(selected)) != len(selected):
            raise ValueError("Need unique layer numbers and one rate per layer")
        if selected != sorted(selected) or any(type(i) is not int or not 1 <= i <= len(self.layers) for i in selected):
            raise ValueError("Layers must be ascending one-based native Encoder layers")
        if any(type(r) not in [int, float] or not 0 < r <= 1 for r in rates):
            raise ValueError("Keep rates must be in (0,1]")
        if any(hasattr(layer, "_ffn_skip_original_forward") for layer in model.model.text_model.layers):
            raise ValueError("Do not combine Encoder pruning with FFN Skip")
        if hasattr(model, "_encoder_token_prune_patch"):
            raise ValueError("Encoder patch already installed")
        if any(type(layer) is not Idefics3EncoderLayer for layer in self.layers):
            raise TypeError("Unreviewed Encoder layer class")
        source = inspect.getsourcefile(Idefics3EncoderLayer)
        with open(source, "rb") as stream:
            source_sha = hashlib.sha256(stream.read()).hexdigest()
        if source_sha != contract["source_sha256"]:
            raise ValueError("Installed Encoder source differs from frozen contract")
        if model.config.vision_config._attn_implementation != "eager" or model.training:
            raise ValueError("Only reviewed eager eval path is supported")
        self.rates = dict(zip(selected, rates))
        self.bypass = not selected or all(r == 1 for r in rates)
        self.original_encoder = self.encoder.forward
        self.original_connector = self.connector.forward
        self.original_layers = {i: self.layers[i-1].forward for i in selected}
        self.calls, self.state = [], None
        model._encoder_token_prune_patch = self
        owner = self
        def encoder_forward(this, inputs_embeds, attention_mask=None):
            return owner.encode(inputs_embeds, attention_mask)
        def connector_forward(this, image_hidden_states):
            return owner.connect(image_hidden_states)
        self.encoder.forward = MethodType(encoder_forward, self.encoder)
        self.connector.forward = MethodType(connector_forward, self.connector)
        for number in selected:
            def layer_forward(this, hidden_states, attention_mask, _number=number, **kwargs):
                return owner.prune_layer(_number, this, hidden_states, attention_mask, **kwargs)
            self.layers[number-1].forward = MethodType(layer_forward, self.layers[number-1])

    def encode(self, hidden, mask):
        report = dict(installed=True, layers_1based=list(self.rates), rates=list(self.rates.values()),
                      bypass=self.bypass, score_calls=0, scatter_calls=0, patched_layer_calls=0,
                      original_encoder_input=list(hidden.shape), stages=[], layer_lengths=[])
        self.calls.append(report)
        self.report = report
        if self.bypass:
            self.state = None
            result = self.original_encoder(hidden, mask)
            report["encoder_output"] = list(result.last_hidden_state.shape)
            return result
        bsz, n, _ = hidden.shape
        if mask is None:
            valid = torch.ones((bsz, n), dtype=torch.bool, device=hidden.device)
        else:
            if list(mask.shape) != [bsz, 1, n, n]:
                raise ValueError("Unexpected native additive mask dimensions")
            valid = mask[:, 0, 0, :] == 0
        if not valid.any(1).all():
            raise ValueError("An image block has no valid patch tokens")
        self.state = dict(valid=valid, positions=torch.arange(n, device=hidden.device).expand(bsz, -1).clone(), original_length=n)
        report["initial_valid_per_block"] = valid.sum(1).tolist()
        for number, layer in enumerate(self.layers, 1):
            old = hidden.shape[1]
            before_mask = list(mask.shape) if mask is not None else None
            hidden = layer(hidden, mask)
            report["layer_lengths"].append(dict(layer=number, attention_physical_length=old,
                                                ffn_physical_length=hidden.shape[1], mask_before=before_mask))
            if number in self.rates and self.rates[number] < 1:
                mask = _prepare_4d_attention_mask(self.state["valid"], hidden.dtype)
                report["stages"][-1]["next_mask_shape"] = list(mask.shape)
        report["encoder_output"] = list(hidden.shape)
        return BaseModelOutput(last_hidden_state=hidden)

    def prune_layer(self, number, layer, hidden, mask, **kwargs):
        self.report["patched_layer_calls"] += 1
        if self.rates[number] == 1:
            return self.original_layers[number](hidden, mask, **kwargs)
        residual = hidden
        attn, weights = layer.self_attn(hidden_states=layer.layer_norm1(hidden), attention_mask=mask, **kwargs)
        hidden = residual + attn
        valid, positions = self.state["valid"], self.state["positions"]
        scores = attention_scores(weights, valid)
        self.report["score_calls"] += 1
        hidden, new_positions, new_valid, selected, ks = select_tokens(hidden, positions, valid, scores, self.rates[number])
        event = dict(layer=number, attention_shape=list(weights.shape),
                     effective_before=valid.sum(1).tolist(), effective_k=ks, physical_k=hidden.shape[1],
                     mask_shape_before=list(mask.shape) if mask is not None else None,
                     score_dtype=str(scores.dtype), fresh_attention_score_call=self.report["score_calls"])
        if self.trace:
            event.update(original_positions_before=positions.tolist(), original_positions_after=new_positions.tolist(),
                         selected_current_indices=selected, valid_after=new_valid.tolist(),
                         scores_sha256=hashlib.sha256(scores.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        self.report["stages"].append(event)
        self.state.update(valid=new_valid, positions=new_positions)
        del weights, scores
        # Both the FFN input and its residual now have the selected physical length.
        residual = hidden
        hidden = layer.mlp(layer.layer_norm2(hidden))
        return residual + hidden

    def connect(self, sparse_postnorm):
        self.report["postnorm_output"] = list(sparse_postnorm.shape)
        if self.bypass:
            self.report["connector_core_input"] = list(sparse_postnorm.shape)
            return self.original_connector(sparse_postnorm)
        positions, valid = self.state["positions"], self.state["valid"]
        grid, occupied = restore_grid(sparse_postnorm, positions, valid, self.state["original_length"])
        self.report["scatter_calls"] += 1
        self.report["connector_core_input"] = list(grid.shape)
        self.report["dropped_positions_exact_zero"] = not torch.count_nonzero(grid[~occupied]).item()
        for b in range(grid.shape[0]):
            if not torch.equal(grid[b, positions[b, valid[b]]], sparse_postnorm[b, valid[b]]):
                raise AssertionError("Retained postnorm scatter differs")
        self.report["retained_scatter_exact"] = True
        if self.trace:
            self.report["final_original_positions"] = positions.tolist()
            self.report["final_valid_mask"] = valid.tolist()
        return self.original_connector(grid)

    def restore(self):
        self.encoder.forward, self.connector.forward = self.original_encoder, self.original_connector
        for i, original in self.original_layers.items():
            self.layers[i-1].forward = original
        del self.model._encoder_token_prune_patch

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
