"""Fixed same-question context diagnosis, with atomic records and strict resume."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import traceback

from evaluate_baseline import atomic, append, digest, now, sha
from day2_runtime import ROOT, assert_environment, current_environment

CODE = ["scripts/diagnose_context.py", "scripts/context_runtime.py", "scripts/run_original.py",
        "scripts/check_scores.py", "scripts/vendor/mmf_m4c_evaluators.py",
        "scripts/day2_runtime.py", "scripts/evaluate_baseline.py"]


def summarize(output, rows, binding_id):
    pairs = []
    for row in rows:
        records = [json.loads((output / name / f"row_{row['row_index']:05d}.json").read_text())
                   for name in ("no_hint", "hint")]
        for r in records:
            assert r["status"] == "success" and r["binding_id"] == binding_id
            assert r["stable_id"] == row["stable_id"]
        old, new = records
        blank = not row["hint"].strip()
        checks = dict(prompt_equal=old["prompt"] == new["prompt"],
                      input_ids_equal=old["input_ids"] == new["input_ids"],
                      generation_equal=old["new_token_ids"] == new["new_token_ids"])
        if blank:
            assert all(checks.values()), (row["row_index"], "empty-hint control differs", checks)
        else:
            assert not checks["prompt_equal"] and not checks["input_ids_equal"]
            assert new["question_for_model"] == "Hint: " + row["hint"] + "\n" + old["question_for_model"]
        pairs.append(dict(row_index=row["row_index"], hint_nonempty=not blank, checks=checks,
                          old_score=old["score"], new_score=new["score"],
                          old_raw=old["raw_generated_text"], new_raw=new["raw_generated_text"],
                          old_parse=old["parse_status"], new_parse=new["parse_status"],
                          old_length=old["input_length"], new_length=new["input_length"]))
    groups = {}
    for name, subset in [("all", pairs), ("nonempty", [p for p in pairs if p["hint_nonempty"]]),
                         ("empty_control", [p for p in pairs if not p["hint_nonempty"]])]:
        groups[name] = dict(n=len(subset), old_correct=sum(p["old_score"] for p in subset),
                            new_correct=sum(p["new_score"] for p in subset),
                            improved=sum(p["new_score"] > p["old_score"] for p in subset),
                            regressed=sum(p["new_score"] < p["old_score"] for p in subset),
                            unchanged=sum(p["new_score"] == p["old_score"] for p in subset))
    report = dict(status="passed", binding_id=binding_id, groups=groups, pairs=pairs,
                  scope="16 fixed diagnostic questions, 32 actual generations; not formal 256 accuracy or official replication",
                  protocol_choice="semantic decision predates generation; independent of score direction")
    atomic(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / "data/manifests/scienceqa_img_validation_256.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 256 and rows == sorted(rows, key=lambda r: r["row_index"])
    selected = sorted([r for r in rows if r["hint"].strip()][:12] +
                      [r for r in rows if not r["hint"].strip()][:4], key=lambda r: r["row_index"])
    protocol = json.loads(args.protocol.read_text())
    from context_runtime import validate_protocol, build_question, load_context_model
    from run_original import run_sample, build_question as build_v1
    validate_protocol(protocol)
    old = json.loads((ROOT / "configs/eval_protocol.json").read_text())
    # All 512 inputs checked before generating, including label-field independence.
    input_checks = {}
    for key, manifest in [("scienceqa", path), ("textvqa", ROOT / "data/manifests/textvqa_validation_256.jsonl")]:
        items = [json.loads(line) for line in manifest.read_text().splitlines()]
        for item in items:
            before, after = build_v1(key, item, old), build_question(key, item, protocol)
            expected = "Hint: " + item["hint"] + "\n" + before if key == "scienceqa" and item["hint"].strip() else before
            assert after == expected
            poisoned = dict(item, answer="FORBIDDEN_ANSWER", lecture="FORBIDDEN_LECTURE", solution="FORBIDDEN_SOLUTION")
            assert build_question(key, poisoned, protocol) == after
        input_checks[key] = dict(count=len(items), exact_expected_delta=True, label_field_independent=True)
    binding = dict(environment_id=assert_environment(), environment=current_environment(),
                   protocol_sha256=sha(args.protocol), old_protocol_sha256=sha(ROOT / "configs/eval_protocol.json"),
                   manifest_sha256=sha(path), selected_rows=[r["row_index"] for r in selected],
                   selection="first 12 nonempty + first 4 empty hint, ascending original manifest row_index; score blind",
                   code_sha256={f: sha(ROOT / f) for f in CODE})
    binding_id = digest(binding)
    metadata = output / "run.json"
    if metadata.exists():
        prior = json.loads(metadata.read_text())
        if not args.resume or prior["binding"] != binding:
            raise ValueError("Existing diagnostic requires --resume and identical binding")
    else:
        if args.resume:
            raise ValueError("Nothing to resume")
        atomic(metadata, dict(created_at=now(), binding=binding, binding_id=binding_id))
        atomic(output / "selection.json", selected)
        atomic(output / "input_checks.json", input_checks)
        shutil.copy2(args.protocol, output / "protocol_draft_snapshot.json")
        for f in CODE:
            dest = output / "code_snapshot" / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
    event_file = output / "events.jsonl"
    append(event_file, dict(event="invocation_start", at=now(), argv=sys.argv))
    pending = [(r, name) for r in selected for name in ("no_hint", "hint")
               if not (output / name / f"row_{r['row_index']:05d}.json").exists()]
    if pending:
        model, processor = load_context_model(output / "runtime", protocol)
        append(event_file, dict(event="model_loaded_once", at=now()))
        for row, name in pending:
            target = output / name / f"row_{row['row_index']:05d}.json"
            p = old if name == "no_hint" else protocol
            try:
                image = ROOT / row["image_file"]
                assert sha(image) == row["image_sha256"]
                question = build_question("scienceqa", row, p)
                record = run_sample(model, processor, "scienceqa", image, question, sample=row,
                                    output_dir=output / "runtime", trace_enabled=False,
                                    protocol=p, persist=False, allow_empty=True)
                record.update(status="success", stable_id=row["stable_id"], row_index=row["row_index"],
                              variant=name, binding_id=binding_id, protocol_sha256=binding["old_protocol_sha256"] if name == "no_hint" else binding["protocol_sha256"])
                atomic(target, record)
                append(event_file, dict(event="committed", row_index=row["row_index"], variant=name, at=now()))
            except Exception:
                atomic(target, dict(status="error", binding_id=binding_id, stable_id=row["stable_id"], traceback=traceback.format_exc()))
                append(event_file, dict(event="error", row_index=row["row_index"], variant=name, at=now()))
                raise
    report = summarize(output, selected, binding_id)
    append(event_file, dict(event="complete", at=now(), groups=report["groups"]))
    print(json.dumps(report["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
