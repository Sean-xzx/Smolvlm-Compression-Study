"""Versioned context adapter; reuse original HF loading/generation and scorers.

The v1 loader couples model checks with a no-hint prompt assertion. For context
v2 we call its protocol-independent loading path and explicitly validate every
inherited setting here. No false v1 protocol is passed to the loader.
"""
import json
from pathlib import Path

from evaluate_baseline import sha
from run_original import ROOT, MODEL_ID, REVISION, build_question as build_v1_question
from run_original import load_original_model as load_hf_model

V1_PATH = ROOT / "configs/eval_protocol.json"
V1_SHA = "224e08d1c152ee480775ce636506d35b5be5796b585de768067d446f3f758080"
CONTEXT_RULE = {
    "field": "hint", "nonempty_test": "bool(hint.strip())",
    "prefix": "Hint: ", "separator": "\n", "preserve_raw_hint": True,
    "empty_hint": "byte-identical to v1",
    "allowed_text_fields": ["hint", "question", "choices"],
    "excluded_fields": ["answer", "lecture", "solution"],
}


def validate_protocol(protocol):
    if sha(V1_PATH) != V1_SHA:
        raise ValueError("Historical v1 protocol changed")
    original = json.loads(V1_PATH.read_text())
    if protocol == original:
        return "v1"
    # All scientific settings except the explicitly specified hint rule inherit v1.
    metadata = {"protocol_version", "status", "configuration", "frozen_at",
                "frozen_from_draft_sha256", "preflight_evidence", "freeze_note"}
    expected = {k: v for k, v in original.items() if k not in metadata}
    actual = {k: v for k, v in protocol.items()
              if k not in metadata | {"context_rule", "context_decision"}}
    expected["prompts"]["scienceqa"]["use_hint"] = True
    if actual != expected or protocol.get("context_rule") != CONTEXT_RULE:
        raise ValueError("Unsupported scientific change outside the hint context delta")
    if protocol["protocol_version"] != "context-v2" or protocol["status"] not in {"draft", "frozen"}:
        raise ValueError("Unsupported context protocol version/status")
    if not protocol.get("context_decision"):
        raise ValueError("Context semantic decision required")
    return "context-v2"


def build_question(key, sample, protocol):
    validate_protocol(protocol)
    # The text builder never receives answer/lecture/solution or scoring metadata.
    safe = {"question": sample["question"]}
    if key == "scienceqa":
        safe["choices"] = sample["choices"]
    question = build_v1_question(key, safe, protocol)
    if key == "scienceqa" and protocol["prompts"][key]["use_hint"]:
        hint = sample["hint"]
        if not isinstance(hint, str):
            raise ValueError("ScienceQA hint must be a string")
        if hint.strip():
            question = "Hint: " + hint + "\n" + question
    return question


def load_context_model(output_dir, protocol):
    version = validate_protocol(protocol)
    if version == "v1":
        return load_hf_model(output_dir, protocol)
    # These checks run against the real context protocol, before loading.
    assert protocol["model"]["repo_id"] == MODEL_ID
    assert protocol["model"]["revision"] == REVISION
    assert protocol["scoring"]["code_sha256"] == sha(ROOT / "scripts/check_scores.py")
    assert protocol["scoring"]["mmf_sha256"] == sha(ROOT / "scripts/vendor/mmf_m4c_evaluators.py")
    model, processor = load_hf_model(output_dir, protocol=None)
    import torch
    assert str(model.device) == protocol["device"]
    assert str(model.dtype) == "torch." + protocol["dtype"]
    assert not model.training and protocol["eval"] and protocol["inference_mode"]
    assert protocol["batch_size"] == 1 and protocol["generation_seed"] == 0
    assert torch.are_deterministic_algorithms_enabled() == protocol["deterministic_algorithms"]
    assert torch.backends.cuda.matmul.allow_tf32 == protocol["allow_tf32"]
    assert torch.backends.cudnn.allow_tf32 == protocol["allow_tf32"]
    assert model.config._commit_hash == protocol["model"]["revision"]
    for config in (model.config, model.config.vision_config, model.config.text_config):
        assert config._attn_implementation == protocol["attention_backend"]
    actual = processor.image_processor.to_dict()
    for key, value in protocol["preprocessing"].items():
        if key == "processor_class":
            assert type(processor).__name__ == value
        else:
            assert actual.get(key) == value, (key, actual.get(key), value)
    from evaluate_baseline import atomic
    atomic(Path(output_dir) / "context_validation.json", {
        "protocol_version": version, "validated_protocol": protocol,
        "loading_path": "original load_original_model(protocol=None), independently validated here",
        "checks": ["exact v1 scientific settings except hint", "model revision and original weight SHA",
                   "all preprocessing fields", "dtype/device/eager/eval/determinism/TF32/seed",
                   "unchanged scorer SHA; actual run_sample receives real protocol generation settings"],
    })
    return model, processor
