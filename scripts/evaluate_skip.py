"""Thin Skip entry; reuses original loading/generation/scoring and record helpers.

Manifest validation / serial loop adapted from evaluate_baseline.py. That file
and its original four source bindings remain untouched.
"""
import argparse
import fcntl
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from evaluate_baseline import ROOT, CODE as BASE_CODE, sha, digest, now, atomic, append, record_path, summarize

CODE = BASE_CODE + ["scripts/ffn_skip.py", "scripts/evaluate_skip.py", "scripts/verify_skip.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--skip-config", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=["preflight", "evaluation"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    output = args.output_dir.resolve()
    allowed = [ROOT / "evidence/day1_evening", ROOT / "results"]
    if not any(base in output.parents for base in allowed) or output == ROOT / "results/baseline_v1_20261029" or ROOT / "results/baseline_v1_20261029" in output.parents:
        parser.error("Use a new evening/Skip output directory, never baseline or morning history")
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.skip_config.read_text())
    assert protocol["status"] == config["status"] == "frozen"
    assert config["configuration"] == "llm_ffn_skip"
    assert config["baseline_protocol_sha256"] == sha(args.protocol)
    assert config["indices_0based"] == [i - 1 for i in config["layers_1based"]]
    assert config["skip_scope"] == ["post_attention_layernorm", "mlp"]
    manifests, manifest_info = {}, {}
    for path in args.manifest:
        items = [json.loads(line) for line in path.read_text().splitlines()]
        key = items[0]["dataset_key"]
        assert key not in manifests
        spec = protocol["datasets"][key]
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert sha(path) == meta["manifest_sha256"]
        assert meta["binding"]["dataset"] == spec and meta["binding"]["sampling"] == protocol["sampling"]
        assert len(items) == protocol["sampling"]["count"]
        assert len({i["stable_id"] for i in items}) == len(items)
        assert [i["row_index"] for i in items] == meta["binding"]["original_indices"]
        assert [i["row_index"] for i in items] == sorted(random.Random(protocol["sampling"]["seed"]).sample(range(spec["population_size"]), protocol["sampling"]["count"]))
        for item in items:
            assert item["revision"] == spec["revision"] and item["dataset"] == spec["repo_id"]
            assert item["split"] == spec["split"] and item["dataset_key"] == key
        manifests[key] = items
        manifest_info[key] = dict(path=str(path.resolve()), sha256=sha(path), count=len(items))
    # Locate installed sources without importing torch/model code for summaries.
    package = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
    sources = [package / "models/llama/modeling_llama.py", package / "models/idefics3/modeling_idefics3.py"]
    source_sha = {str(p): sha(p) for p in sources}
    assert source_sha[str(sources[0])] == config["llama_source_sha256"]
    binding = dict(role="skip_" + args.role, configuration=config["configuration"],
                   protocol_sha256=sha(args.protocol), skip_config_sha256=sha(args.skip_config),
                   manifests=manifest_info, python=sys.version,
                   code_sha256={f: sha(ROOT / f) for f in CODE}, upstream_source_sha256=source_sha,
                   dependencies={k: importlib.metadata.version(k) for k in ["torch", "torchvision", "transformers", "datasets", "Pillow", "huggingface_hub", "safetensors"]})
    binding_id = digest(binding)
    validation_evidence = {}
    for stage in ["empty", "enabled", "profiler", "parameters"]:
        proof_path = args.validation_dir / f"{stage}.json"
        proof = json.loads(proof_path.read_text())
        assert proof["passed"] and proof["binding"]["protocol_sha256"] == binding["protocol_sha256"]
        assert proof["binding"]["skip_config_sha256"] == binding["skip_config_sha256"]
        for file, expected in proof["binding"]["code_sha256"].items():
            assert sha(ROOT/file) == expected, "Verification used another source version"
        validation_evidence[stage] = dict(path=str(proof_path.resolve()), sha256=sha(proof_path))
    if args.role == "evaluation":
        if args.preflight_dir is None:
            raise ValueError("Formal Skip evaluation requires the passed 8-per-dataset preflight")
        prior_run = json.loads((args.preflight_dir/"run.json").read_text())
        prior = json.loads((args.preflight_dir/"verification.json").read_text())
        assert prior["passed"] and all(d["count"] == 8 for d in prior["datasets"].values())
        for field in ["protocol_sha256", "skip_config_sha256", "manifests", "code_sha256", "upstream_source_sha256", "dependencies", "python"]:
            assert prior_run["binding"][field] == binding[field], "Preflight binding differs"
        validation_evidence["preflight"] = dict(path=str(args.preflight_dir.resolve()), verification_sha256=sha(args.preflight_dir/"verification.json"))
    binding["validation_evidence"] = validation_evidence
    binding_id = digest(binding)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta_path = output / "run.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text())["binding"] != binding:
            raise ValueError("Skip run binding mismatch: protocol/config/code/source/software/manifest changed")
        if not args.resume and not args.summarize_only:
            raise ValueError("Existing Skip run requires --resume")
    else:
        if args.resume or args.summarize_only:
            raise ValueError("No existing run")
        if any(output.glob("*/samples/*.json")):
            raise ValueError("Unbound sample records already exist")
        atomic(meta_path, dict(run_id=output.name, created_at=now(), binding=binding, binding_id=binding_id))
        shutil.copy2(args.protocol, output / "protocol_snapshot.json")
        shutil.copy2(args.skip_config, output / "skip_config_snapshot.json")
        for key, info in manifest_info.items():
            path = Path(info["path"])
            (output / "manifest_snapshot").mkdir(exist_ok=True)
            shutil.copy2(path, output / "manifest_snapshot" / f"{key}.jsonl")
            shutil.copy2(path.with_suffix(".meta.json"), output / "manifest_snapshot" / f"{key}.meta.json")
        for f in CODE:
            dest = output / "code_snapshot" / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
        (output / "upstream_snapshot").mkdir()
        for path in sources:
            shutil.copy2(path, output / "upstream_snapshot" / path.name)
    for key in manifests:
        (output / key / "samples").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(json.dumps(summarize(output, manifests, binding_id, args.limit, binding["role"]), indent=2))
        return
    events = output / "events.jsonl"
    append(events, dict(event="invocation_start", started_at=now(), command=sys.argv, resume=args.resume, limit=args.limit))
    started = time.monotonic()
    pending = []
    for key, items in manifests.items():
        for item in items[:args.limit] if args.limit else items:
            path = record_path(output, key, item)
            if path.exists():
                record = json.loads(path.read_text())
                assert record["binding_id"] == binding_id and record["stable_id"] == item["stable_id"]
                if record["status"] == "success":
                    append(events, dict(event="skip_committed", stable_id=item["stable_id"], at=now()))
                    continue
                if not args.retry_errors:
                    raise ValueError("Diagnose saved error and use --retry-errors")
            pending.append((key, item))
    if pending:
        from run_original import load_original_model, run_sample, build_question
        from ffn_skip import SkipPatch
        append(events, dict(event="model_load_start", at=now()))
        model, processor = load_original_model(output / "runtime", protocol)
        patch = SkipPatch(model, config["layers_1based"], config["llama_source_sha256"])
        atomic(output / "runtime/skip_applied.json", dict(configuration=config["configuration"], layers_1based=patch.layers_1based, indices_0based=config["indices_0based"], binding_id=binding_id))
        append(events, dict(event="model_load_complete", at=now(), model_loads_this_invocation=1, layers_1based=patch.layers_1based))
        try:
            for key, item in pending:
                wall = time.monotonic()
                append(events, dict(event="generation_start", stable_id=item["stable_id"], at=now()))
                try:
                    image = ROOT / item["image_file"]
                    if sha(image) != item["image_sha256"]:
                        raise ValueError("Input image checksum mismatch")
                    question = build_question(key, item, protocol)
                    record = run_sample(model, processor, key, image, question, sample=item,
                                        output_dir=output / "runtime", trace_enabled=False,
                                        protocol=protocol, persist=False, allow_empty=True)
                    record.update(status="success", inference_status="ok", stable_id=item["stable_id"],
                                  binding_id=binding_id, row_index=item["row_index"],
                                  configuration=config["configuration"], skip_layers_1based=patch.layers_1based,
                                  scheduling_sample_wall_seconds=time.monotonic()-wall)
                    atomic(record_path(output, key, item), record)
                    append(events, dict(event="record_committed", stable_id=item["stable_id"], at=now(), parse_status=record["parse_status"], score=record["score"]))
                except Exception as exc:
                    error = dict(status="error", inference_status="exception", stable_id=item["stable_id"],
                                 binding_id=binding_id, sample=item, row_index=item["row_index"], score=None,
                                 error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), recorded_at=now())
                    atomic(record_path(output, key, item), error)
                    append(events, dict(event="execution_error", **error))
                    summarize(output, manifests, binding_id, args.limit, binding["role"])
                    raise
        finally:
            patch.restore()
    report = summarize(output, manifests, binding_id, args.limit, binding["role"])
    append(events, dict(event="invocation_complete", at=now(), new_inferences=len(pending), scheduling_wall_seconds=time.monotonic()-started, complete_requested=all(d["complete_requested"] for d in report.values())))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
