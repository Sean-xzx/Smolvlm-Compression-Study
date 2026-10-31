"""One thin serial quality entry for baseline, Skip and four Encoder variants.

Uses existing HF loading/generate/scorers and atomic record/summary helpers.
No performance interpretation is attached to these instrumented quality runs.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from evaluate_baseline import atomic, append, now, sha, digest, record_path, summarize
from experiment_runtime import (ROOT, BASE, PROTOCOL, NAMES, CORE_CODE, MANIFESTS, configs,
                                scientific_binding, patch_for, gpu_lock, clean_model, method_signature)

ENTRY_CODE = ["scripts/evaluate_context.py"]


def manifests_for(protocol):
    manifests = {}
    for path in MANIFESTS:
        items = [json.loads(l) for l in path.read_text().splitlines()]
        key = items[0]["dataset_key"]; spec = protocol["datasets"][key]
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert sha(path) == meta["manifest_sha256"]
        assert meta["binding"]["dataset"] == spec and meta["binding"]["sampling"] == protocol["sampling"]
        assert len(items) == protocol["sampling"]["count"] == 256
        assert len({i["stable_id"] for i in items}) == len(items)
        indices = [i["row_index"] for i in items]
        assert indices == meta["binding"]["original_indices"]
        assert indices == sorted(random.Random(protocol["sampling"]["seed"]).sample(range(spec["population_size"]),256))
        for item in items:
            assert item["revision"]==spec["revision"] and item["dataset"]==spec["repo_id"]
            assert item["split"]==spec["split"] and item["dataset_key"]==key
        manifests[key] = items
    return manifests


def common_binding(name):
    scientific = scientific_binding()
    proof_path = BASE / "b2/proof.json"
    proof = json.loads(proof_path.read_text())
    assert proof["passed"]
    assert {k:v for k,v in proof["binding"].items() if k!="verification_code_sha256"} == scientific
    for f,h in proof["binding"]["verification_code_sha256"].items():
        assert sha(ROOT/f)==h, "B2 verification source changed"
    return dict(scientific=scientific,configuration=name,
                entry_code_sha256={f:sha(ROOT/f) for f in ENTRY_CODE},
                proof=dict(path=str(proof_path.relative_to(ROOT)),sha256=sha(proof_path)),
                record_policy="atomic JSON authoritative; raw/IDs/reference/parser/score/lengths/errors retained",
                trace_policy="Encoder trace=False; stage counts/physical lengths/zero-scatter checks retained, CPU synchronization overhead remains; no latency claim")


def check_formal_gates(name, common):
    preflight = BASE / f"b3/preflight/{name}"
    report = json.loads((preflight/"verification.json").read_text())
    assert report["passed"] and report["common_binding_id"] == digest(common)
    assert all(d["count"]==8 for d in report["datasets"].values())
    gates = BASE / "b3/verification.json"
    assert json.loads(gates.read_text())["passed"]
    profile = BASE / "b4/report.json"
    assert json.loads(profile.read_text())["passed"]
    freeze = BASE / "b5/source_freeze.json"
    frozen = json.loads(freeze.read_text())
    assert frozen["status"]=="frozen"
    for f,h in frozen["files_sha256"].items():
        if sha(ROOT/f)!=h: raise ValueError("Formal frozen source/config/proof changed: "+f)
    return dict(preflight_sha256=sha(preflight/"verification.json"),binding_checks_sha256=sha(gates),
                profiler_report_sha256=sha(profile),source_freeze_sha256=sha(freeze))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",choices=NAMES,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--role",choices=["preflight","evaluation"],required=True)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-errors",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    if args.limit is not None and not 0<args.limit<=256: parser.error("limit must be 1..256")
    if args.role=="evaluation" and args.limit is not None: parser.error("Formal evaluation must request full manifests")
    if sha(args.protocol)!=sha(PROTOCOL): raise ValueError("Only the independently frozen context-v2 protocol is supported")
    protocol=json.loads(args.protocol.read_text());assert protocol["status"]=="frozen"
    manifests=manifests_for(protocol)
    common=common_binding(args.configuration)
    gates=check_formal_gates(args.configuration,common) if args.role=="evaluation" else None
    binding=dict(common=common,role=args.role,formal_gates=gates)
    binding_id=digest(binding)
    output=args.output_dir.resolve()
    if args.role=="evaluation":
        if output.parent!=ROOT/"results" or not output.name.startswith("context_v2_"):
            parser.error("Formal output must be a new results/context_v2_* directory")
    elif BASE/"b3" not in output.parents:
        parser.error("Preflight output must be under this batch b3")
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/".run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=output/"run.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["binding"]!=binding:
            raise ValueError("Run binding mismatch: configuration/protocol/source/environment/manifest/proof/gates changed")
        if not args.resume and not args.summarize_only: raise ValueError("Existing run requires --resume or --summarize-only")
    else:
        if args.resume or args.summarize_only: raise ValueError("No bound run to resume/aggregate")
        if any(output.glob("*/samples/*.json")):raise ValueError("Unbound sample files present")
        atomic(metadata,dict(run_id=output.name,created_at=now(),binding=binding,binding_id=binding_id,common_binding_id=digest(common)))
        files=CORE_CODE+ENTRY_CODE+[str(PROTOCOL.relative_to(ROOT))]+list(common["scientific"]["configurations_sha256"])+list(common["scientific"]["manifests_sha256"])
        for f in set(files):
            dest=output/"source_snapshot"/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
        atomic(output/"environment.json",common["scientific"]["environment"])
    for key in manifests: (output/key/"samples").mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        print(json.dumps(summarize(output,manifests,binding_id,args.limit,args.configuration),indent=2));return
    events=output/"events.jsonl";started=time.monotonic()
    append(events,dict(event="invocation_start",at=now(),command=sys.argv,resume=args.resume,limit=args.limit))
    pending=[]
    for key,items in manifests.items():
        for item in items[:args.limit] if args.limit else items:
            p=record_path(output,key,item)
            if p.exists():
                r=json.loads(p.read_text());assert r["binding_id"]==binding_id and r["stable_id"]==item["stable_id"]
                if r["status"]=="success":
                    append(events,dict(event="skip_committed",at=now(),stable_id=item["stable_id"]));continue
                if not args.retry_errors:raise ValueError("Persisted error requires diagnosis and --retry-errors")
                dest=output/"error_history"/key/(p.stem+"_"+str(time.time_ns())+".json")
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            pending.append((key,item))
    if pending:
        from context_runtime import load_context_model,build_question
        from run_original import run_sample
        with gpu_lock():
            append(events,dict(event="model_load_start",at=now()))
            model,processor=load_context_model(output/"runtime",protocol)
            methods=method_signature(model);cfgs,contract=configs()
            append(events,dict(event="model_load_complete",at=now(),model_loads_this_invocation=1,parameter_count=sum(p.numel() for p in model.parameters())))
            try:
                with patch_for(model,args.configuration,cfgs,contract,trace=False) as patch:
                    for key,item in pending:
                        wall=time.monotonic()
                        append(events,dict(event="generation_start",at=now(),stable_id=item["stable_id"]))
                        try:
                            image=ROOT/item["image_file"]
                            if sha(image)!=item["image_sha256"]:raise ValueError("Input image SHA differs from manifest")
                            if hasattr(patch,"calls"):patch.calls.clear()
                            question=build_question(key,item,protocol)
                            record=run_sample(model,processor,key,image,question,sample=item,output_dir=output/"runtime",
                                              trace_enabled=False,protocol=protocol,persist=False,allow_empty=True)
                            record.update(status="success",inference_status="ok",stable_id=item["stable_id"],row_index=item["row_index"],
                                          configuration=args.configuration,binding_id=binding_id,
                                          scheduling_sample_wall_seconds=time.monotonic()-wall)
                            if hasattr(patch,"calls"):
                                assert len(patch.calls)==1 and not patch.calls[0]["bypass"]
                                record["encoder_computation"]=copy.deepcopy(patch.calls[0])
                                record["encoder_dimensions"]=dict(hidden=model.config.vision_config.hidden_size,intermediate=model.config.vision_config.intermediate_size)
                            assert record["question_for_model"]==question
                            atomic(record_path(output,key,item),record)
                            append(events,dict(event="record_committed",at=now(),stable_id=item["stable_id"],score=record["score"],parse_status=record["parse_status"]))
                        except Exception as exc:
                            error=dict(status="error",inference_status="exception",stable_id=item["stable_id"],row_index=item["row_index"],
                                       sample=item,binding_id=binding_id,score=None,error_type=type(exc).__name__,error=str(exc),
                                       traceback=traceback.format_exc(),recorded_at=now())
                            atomic(record_path(output,key,item),error);append(events,dict(event="execution_error",**error))
                            summarize(output,manifests,binding_id,args.limit,args.configuration);raise
            finally:
                clean_model(model);assert method_signature(model)==methods
                append(events,dict(event="patch_restored",at=now()))
    report=summarize(output,manifests,binding_id,args.limit,args.configuration)
    append(events,dict(event="invocation_complete",at=now(),new_inferences=len(pending),scheduling_wall_seconds=time.monotonic()-started,
                       complete_requested=all(d["complete_requested"] for d in report.values())))
    print(json.dumps({k:dict(count=v["inferred_count"],requested=v["requested_target"],complete=v["complete_requested"],mean=v["mean_score"]) for k,v in report.items()},indent=2),flush=True)


if __name__=="__main__":main()
