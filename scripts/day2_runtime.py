"""Small run-level environment/provenance helpers; no model import or mutation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/day2_morning/20261030_topk_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_environment():
    sources = [ROOT / ".venv/lib/python3.10/site-packages/transformers" / f for f in
               ["models/idefics3/modeling_idefics3.py", "models/idefics3/processing_idefics3.py",
                "models/idefics3/image_processing_idefics3.py", "models/llama/modeling_llama.py"]]
    return dict(python=sys.version, executable=sys.executable,
                resolved_executable=str(Path(sys.executable).resolve()),
                python_binary_sha256=sha(Path(sys.executable).resolve()), prefix=sys.prefix,
                dependencies={n: importlib.metadata.version(n) for n in
                              ["torch", "torchvision", "transformers", "datasets", "huggingface_hub", "Pillow", "safetensors", "numpy", "pyarrow"]},
                sources_sha256={str(p): sha(p) for p in sources},
                gpu_driver=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], text=True).strip())


def assert_environment():
    saved = json.loads((EVIDENCE / "environment/environment.json").read_text())
    if current_environment() != saved["fingerprint"]:
        raise ValueError("Current environment differs from the fixed day-2 fingerprint; create a new version")
    return saved["environment_id"]


def proof_binding(config_paths):
    return dict(environment_id=assert_environment(),
                protocol_sha256=sha(ROOT / "configs/eval_protocol.json"),
                contract_sha256=sha(ROOT / "configs/encoder_topk_contract.json"),
                configs_sha256={str(p.relative_to(ROOT)): sha(p) for p in config_paths},
                code_sha256={f: sha(ROOT / f) for f in
                             ["scripts/day2_runtime.py", "scripts/encoder_topk.py", "scripts/verify_encoder_topk.py",
                              "scripts/run_original.py", "scripts/check_scores.py", "scripts/evaluate_baseline.py",
                              "scripts/vendor/mmf_m4c_evaluators.py"]},
                manifests_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                  [ROOT / "data/manifests/textvqa_validation_256.jsonl", ROOT / "data/manifests/scienceqa_img_validation_256.jsonl"]})
