"""B3 controller: fixed eight, real 3->8 resume and controlled rejection checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_baseline import atomic, sha, now, digest
from experiment_runtime import ROOT, BASE, NAMES, PROTOCOL
from evaluate_context import common_binding, manifests_for


def invoke(name, limit=8, resume=False, suffix=None, extra=None, expected=0):
    out=BASE/f"b3/preflight/{name}"
    argv=[sys.executable,"-B","-u",str(ROOT/"scripts/evaluate_context.py"),"--configuration",name,
          "--role","preflight","--output-dir",str(out),"--limit",str(limit)]
    if resume:argv.append("--resume")
    if extra:argv.extend(extra)
    log=BASE/"b3"/(suffix or f"{name}_{limit}.log")
    with log.open("w") as stream:
        result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    if (result.returncode==0)!=(expected==0):
        raise RuntimeError(f"Unexpected exit {result.returncode}; see {log}")
    return dict(command=argv,exit_code=result.returncode,log=str(log.relative_to(ROOT)))


def verify(name, manifests):
    out=BASE/f"b3/preflight/{name}"
    meta=json.loads((out/"run.json").read_text());common=common_binding(name)
    assert meta["common_binding_id"]==digest(common)
    datasets={}
    for key,rows in manifests.items():
        paths=sorted((out/key/"samples").glob("row_*.json"));assert len(paths)==8
        for row,p in zip(rows[:8],paths):
            r=json.loads(p.read_text());assert r["stable_id"]==row["stable_id"] and r["status"]=="success"
            base=json.loads((BASE/f"b3/preflight/baseline/{key}/samples"/p.name).read_text())
            assert r["input_ids"]==base["input_ids"] and r["prompt"]==base["prompt"]
            assert r["image_placeholder_token_count"]==base["image_placeholder_token_count"]
            if name not in {"baseline","skip"}:
                trace=r["encoder_computation"]
                assert trace["dropped_positions_exact_zero"] and trace["retained_scatter_exact"]
                assert len(trace["layer_lengths"])==12 and trace["score_calls"]==(3 if name=="multi" else 1)
                if row in rows[:2]:
                    debug=json.loads((BASE/f"b2/{name}/{key}/{row['row_index']}/{key}_result.json").read_text())
                    assert debug["new_token_ids"]==r["new_token_ids"] and debug["input_ids"]==r["input_ids"]
        datasets[key]=dict(count=8,stable_ids=[r["stable_id"] for r in rows[:8]],input_equal_to_current_baseline=True)
    events=[json.loads(l) for l in (out/"events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["event"]=="record_committed"])==16
    assert len([e for e in events if e["event"]=="patch_restored"])==(2 if name=="baseline" else 1)
    report=dict(passed=True,recorded_at=now(),configuration=name,common_binding_id=digest(common),
                datasets=datasets,committed_records=16,execution_errors=0,
                debug_vs_quality_equal=(name not in {"baseline","skip"}))
    atomic(out/"verification.json",report)
    return report


def main():
    out=BASE/"b3";out.mkdir(parents=True,exist_ok=True)
    if (out/"verification.json").exists():raise ValueError("B3 complete; preserve evidence")
    invocations=[]
    invocations.append(invoke("baseline",limit=3))
    first={str(p.relative_to(ROOT)):dict(sha256=sha(p),mtime_ns=p.stat().st_mtime_ns)
           for p in (out/"preflight/baseline").glob("*/samples/*.json")}
    assert len(first)==6;atomic(out/"resume_before.json",first)
    invocations.append(invoke("baseline",limit=8,resume=True))
    for f,info in first.items():assert sha(ROOT/f)==info["sha256"] and (ROOT/f).stat().st_mtime_ns==info["mtime_ns"]
    manifests=manifests_for(json.loads(PROTOCOL.read_text()))
    reports={"baseline":verify("baseline",manifests)}
    # Reject actual cross-configuration reuse, before any GPU load or sample write.
    dest=out/"preflight/baseline"
    invocations.append(invoke("skip",resume=True,suffix="negative_configuration.log",extra=["--output-dir",str(dest)],expected=1))
    invocations.append(invoke("baseline",resume=True,suffix="negative_protocol.log",extra=["--protocol",str(ROOT/"configs/eval_protocol.json")],expected=1))
    source=ROOT/"scripts/evaluate_context.py";saved=source.read_bytes();stat=source.stat()
    try:
        source.write_bytes(saved+b"\n# controlled source-binding rejection probe\n")
        invocations.append(invoke("baseline",resume=True,suffix="negative_source.log",expected=1))
    finally:
        source.write_bytes(saved);os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert source.read_bytes()==saved
    for f,info in first.items():assert sha(ROOT/f)==info["sha256"] and (ROOT/f).stat().st_mtime_ns==info["mtime_ns"]
    for name in NAMES[1:]:
        invocations.append(invoke(name))
        reports[name]=verify(name,manifests)
    atomic(out/"verification.json",dict(passed=True,recorded_at=now(),reports=reports,invocations=invocations,
        resume_3_to_8=dict(first_six_records_sha_and_mtime_unchanged=True,committed_total=16),
        negative_checks=dict(configuration_rejected=True,protocol_rejected=True,actual_source_change_rejected=True),
        scope="Each of 6 configurations, first 8 rows of each fixed manifest; no formal records reused"))
    print("B3 passed",flush=True)


if __name__=="__main__":main()
