"""Thin serial baseline loop around existing HF inference and upstream scoring.

Atomic per-sample JSON files are authoritative. JSONL and summaries are rebuilt
from these records, including after interruption; successful IDs are never retried.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = ["scripts/evaluate_baseline.py", "scripts/run_original.py", "scripts/check_scores.py",
        "scripts/vendor/mmf_m4c_evaluators.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record_path(output, key, item):
    return output / key / "samples" / f"row_{item['row_index']:05d}.json"


def summarize(output, manifests, binding_id, limit, role):
    summaries = {}
    for key, items in manifests.items():
        requested = items[:limit] if limit else items
        records = []
        for item in items:
            path = record_path(output, key, item)
            if path.exists():
                record = json.loads(path.read_text())
                assert record["stable_id"] == item["stable_id"] and record["binding_id"] == binding_id
                assert record["sample"]["image_sha256"] == item["image_sha256"]
                records.append(record)
        assert len({r["stable_id"] for r in records}) == len(records)
        good = [r for r in records if r["status"] == "success"]
        by_id = {r["stable_id"]: r for r in records}
        requested_good = [by_id[i["stable_id"]] for i in requested
                          if i["stable_id"] in by_id and by_id[i["stable_id"]]["status"] == "success"]
        complete = len(good) == len(items)
        parse_counts = dict(Counter(r["parse_status"] for r in good))
        lengths = [r["generated_length"] for r in good]
        result = dict(dataset=key, role=role, manifest_target=len(items), requested_target=len(requested),
                      inferred_count=len(good), remaining_manifest=len(items)-len(good),
                      execution_errors=sum(r["status"] == "error" for r in records),
                      missing_requested=len(requested)-len(requested_good),
                      complete_requested=len(requested_good) == len(requested), complete_manifest=complete,
                      mean_score=sum(r["score"] for r in good)/len(items) if complete else None,
                      partial_mean_score=sum(r["score"] for r in good)/len(good) if good else None,
                      parse_status_counts=parse_counts,
                      parse_failure_rate=sum(r["parse_status"] == "parse_failure" for r in good)/len(good) if good else None,
                      empty_output_count=sum(r["parse_status"] == "empty_output" for r in good),
                      limit_reached_count=sum(r["reached_length_limit"] for r in good),
                      truncation_count=sum(r.get("possibly_truncated", r["reached_length_limit"]) for r in good),
                      truncation_rate=sum(r.get("possibly_truncated", r["reached_length_limit"]) for r in good)/len(good) if good else None,
                      generated_length=dict(min=min(lengths), max=max(lengths), mean=sum(lengths)/len(lengths)) if lengths else None,
                      binding_id=binding_id, aggregation="saved records only; no generate or rescore")
        atomic(output / key / "summary.json", result)
        temp = output / key / "predictions.jsonl.tmp"
        with temp.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(output / key / "predictions.jsonl")
        summaries[key] = result
    atomic(output / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=["preflight", "baseline"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    output = args.output_dir.resolve()
    if output == ROOT / "evidence/day1_morning" or ROOT / "evidence/day1_morning" in output.parents:
        parser.error("Historical morning evidence is read-only")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(args.protocol.read_text())
    if args.role == "baseline" and protocol["status"] != "frozen":
        raise ValueError("Formal baseline requires a frozen protocol")
    manifests, manifest_info = {}, {}
    for path in args.manifest:
        items = [json.loads(line) for line in path.read_text().splitlines()]
        key = items[0]["dataset_key"]
        assert key not in manifests
        spec = protocol["datasets"][key]
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert sha(path) == meta["manifest_sha256"]
        assert meta["binding"]["dataset"] == spec
        assert meta["binding"]["sampling"] == protocol["sampling"]
        assert len(items) == protocol["sampling"]["count"]
        assert len({i["stable_id"] for i in items}) == len(items)
        assert [i["row_index"] for i in items] == meta["binding"]["original_indices"]
        assert [i["row_index"] for i in items] == sorted(random.Random(protocol["sampling"]["seed"]).sample(
            range(spec["population_size"]), protocol["sampling"]["count"]))
        for item in items:
            assert item["revision"] == spec["revision"] and item["dataset"] == spec["repo_id"]
            assert item["split"] == spec["split"] and item["dataset_key"] == key
        manifests[key] = items
        manifest_info[key] = dict(path=str(path.resolve()), sha256=sha(path), count=len(items))
    binding = dict(role=args.role, protocol_sha256=sha(args.protocol), manifests=manifest_info,
                   python=sys.version,
                   code_sha256={f: sha(ROOT / f) for f in CODE},
                   dependencies={k: importlib.metadata.version(k) for k in
                                 ["torch", "torchvision", "transformers", "datasets", "Pillow", "huggingface_hub", "safetensors"]})
    binding_id = digest(binding)
    metadata_path = output / "run.json"
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        if prior["binding"] != binding:
            raise ValueError("Resume/aggregation binding mismatch: protocol, manifest, code or software changed")
        if not args.resume and not args.summarize_only:
            raise ValueError("Existing run requires --resume")
    else:
        if args.resume or args.summarize_only:
            raise ValueError("No existing run to resume/aggregate")
        atomic(metadata_path, dict(run_id=output.name, created_at=now(), binding=binding, binding_id=binding_id))
        shutil.copy2(args.protocol, output / "protocol_snapshot.json")
        (output / "manifest_snapshot").mkdir(exist_ok=True)
        for key, info in manifest_info.items():
            path = Path(info["path"])
            shutil.copy2(path, output / "manifest_snapshot" / f"{key}.jsonl")
            shutil.copy2(path.with_suffix(".meta.json"), output / "manifest_snapshot" / f"{key}.meta.json")
        for f in CODE:
            dest = output / "code_snapshot" / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
    for key in manifests:
        (output / key / "samples").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(json.dumps(summarize(output, manifests, binding_id, args.limit, args.role), indent=2))
        return
    invocation = dict(started_at=now(), command=sys.argv, resume=args.resume, limit=args.limit,
                      timing_scope="whole invocation wall clock for scheduling, includes load/I/O; not model latency")
    event_file = output / "events.jsonl"
    append(event_file, dict(event="invocation_start", **invocation))
    started = time.monotonic()
    pending = []
    for key, items in manifests.items():
        for item in items[:args.limit] if args.limit else items:
            path = record_path(output, key, item)
            if path.exists():
                existing = json.loads(path.read_text())
                assert existing["binding_id"] == binding_id and existing["stable_id"] == item["stable_id"]
                if existing["status"] == "success":
                    append(event_file, dict(event="skip_committed", stable_id=item["stable_id"], at=now()))
                    continue
                if not args.retry_errors:
                    raise ValueError("Saved execution error requires diagnosis and --retry-errors")
            pending.append((key, item))
    if pending:
        from run_original import load_original_model, run_sample, build_question
        append(event_file, dict(event="model_load_start", at=now()))
        model, processor = load_original_model(output / "runtime", protocol)
        append(event_file, dict(event="model_load_complete", at=now(), model_loads_this_invocation=1))
        for key, item in pending:
            wall = time.monotonic()
            append(event_file, dict(event="generation_start", stable_id=item["stable_id"], at=now()))
            try:
                image = ROOT / item["image_file"]
                if sha(image) != item["image_sha256"]:
                    raise ValueError("Input image checksum differs from manifest")
                question = build_question(key, item, protocol)
                record = run_sample(model, processor, key, image, question, sample=item,
                                    output_dir=output / "runtime", trace_enabled=False, protocol=protocol,
                                    persist=False, allow_empty=True)
                assert record["question_for_model"] == question
                record.update(status="success", inference_status="ok", stable_id=item["stable_id"],
                              binding_id=binding_id, row_index=item["row_index"],
                              scheduling_sample_wall_seconds=time.monotonic()-wall)
                atomic(record_path(output, key, item), record)
                append(event_file, dict(event="record_committed", stable_id=item["stable_id"], at=now(),
                                       parse_status=record["parse_status"], score=record["score"]))
                print(key, "COMMITTED", item["row_index"], flush=True)
            except Exception as exc:
                error = dict(status="error", inference_status="exception", stable_id=item["stable_id"],
                             binding_id=binding_id, sample=item, row_index=item["row_index"], score=None,
                             error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                             recorded_at=now(), scheduling_sample_wall_seconds=time.monotonic()-wall)
                atomic(record_path(output, key, item), error)
                append(event_file, dict(event="execution_error", **error))
                summarize(output, manifests, binding_id, args.limit, args.role)
                raise
    summary = summarize(output, manifests, binding_id, args.limit, args.role)
    append(event_file, dict(event="invocation_complete", at=now(), new_inferences=len(pending),
                           scheduling_wall_seconds=time.monotonic()-started,
                           complete_requested=all(d["complete_requested"] for d in summary.values())))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
