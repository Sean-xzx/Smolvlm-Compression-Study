"""Shared context-v2 configuration, provenance and patch dispatch; no HF rewrite."""
from contextlib import contextmanager, nullcontext
import fcntl
import json
from pathlib import Path

from day2_runtime import ROOT, assert_environment, current_environment
from evaluate_baseline import sha, digest

BASE = ROOT / "evidence/day2_afternoon/20261030_context_v2"
PROTOCOL = ROOT / "configs/eval_protocol_context_v2.json"
NAMES = ["baseline", "skip", "early", "middle", "late", "multi"]
CORE_CODE = ["scripts/experiment_runtime.py", "scripts/context_runtime.py", "scripts/run_original.py",
             "scripts/encoder_topk.py", "scripts/ffn_skip.py", "scripts/check_scores.py",
             "scripts/vendor/mmf_m4c_evaluators.py", "scripts/evaluate_baseline.py", "scripts/day2_runtime.py"]
MANIFESTS = [ROOT / "data/manifests/textvqa_validation_256.jsonl",
             ROOT / "data/manifests/scienceqa_img_validation_256.jsonl"]


def config_path(name):
    if name not in NAMES:
        raise ValueError("Unknown configuration")
    return ROOT / "configs/context_v2" / (f"encoder_{name}.json" if name not in {"baseline", "skip"} else f"{name}.json")


def configs():
    contract_path = ROOT / "configs/context_v2/encoder_topk_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["base_protocol_sha256"] == sha(PROTOCOL)
    old_contract = json.loads((ROOT / "configs/encoder_topk_contract.json").read_text())
    assert contract["parent_contract_sha256"] == sha(ROOT / "configs/encoder_topk_contract.json")
    for k, v in old_contract.items():
        if k not in {"version", "base_protocol_sha256", "created_at"}:
            assert contract[k] == v, k
    result = {}
    for name in NAMES:
        cfg = json.loads(config_path(name).read_text())
        assert cfg["status"] == "frozen" and cfg["environment_id"] == assert_environment()
        assert cfg.get("base_protocol_sha256", cfg.get("baseline_protocol_sha256")) == sha(PROTOCOL)
        if name not in {"baseline", "skip"}:
            assert cfg["contract_sha256"] == sha(contract_path)
            old = json.loads((ROOT / f"configs/encoder_{name}.json").read_text())
            for k in ["layers_1based", "indices_0based", "keep_rates", "llm_ffn_skip"]:
                assert cfg[k] == old[k]
        elif name == "skip":
            old = json.loads((ROOT / "configs/llm_ffn_skip.json").read_text())
            for k in ["layers_1based", "indices_0based", "skip_scope", "llama_source_sha256"]:
                assert cfg[k] == old[k]
        result[name] = cfg
    return result, contract


def scientific_binding():
    cfgs, _ = configs()
    paths = [PROTOCOL, ROOT / "configs/context_v2/encoder_topk_contract.json"] + [config_path(n) for n in NAMES]
    return dict(environment_id=assert_environment(), environment=current_environment(),
                protocol_sha256=sha(PROTOCOL),
                configurations_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                manifests_sha256={str(p.relative_to(ROOT)): sha(p) for p in MANIFESTS},
                core_code_sha256={f: sha(ROOT / f) for f in CORE_CODE})


def patch_for(model, name, cfgs=None, contract=None, trace=False):
    if cfgs is None:
        cfgs, contract = configs()
    if name == "baseline":
        return nullcontext(None)
    if name == "skip":
        from ffn_skip import SkipPatch
        assert not hasattr(model, "_encoder_token_prune_patch")
        return SkipPatch(model, cfgs[name]["layers_1based"], cfgs[name]["llama_source_sha256"])
    from encoder_topk import EncoderTopKPatch
    return EncoderTopKPatch(model, cfgs[name], contract, trace=trace)


@contextmanager
def gpu_lock():
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / ".gpu.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def clean_model(model):
    assert not hasattr(model, "_encoder_token_prune_patch")
    assert all(not hasattr(l, "_ffn_skip_original_forward") for l in model.model.text_model.layers)


def method_signature(model):
    modules = [model.model.vision_model.encoder, model.model.connector,
               *model.model.vision_model.encoder.layers, *model.model.text_model.layers]
    return [(m.forward, tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in modules]
