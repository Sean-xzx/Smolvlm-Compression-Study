"""Serial six-configuration formal queue; stop on failure, resumable bound runs."""
import json
from pathlib import Path
import subprocess
import sys

from evaluate_baseline import atomic,now,sha
from experiment_runtime import ROOT,BASE,NAMES


def main():
    folder=BASE/"b5";folder.mkdir(parents=True,exist_ok=True)
    status_file=folder/"queue_status.json"
    if not (folder/"source_freeze.json").exists():raise ValueError("Freeze source after gates, before formal queue")
    tasks={}
    for name in NAMES:
        out=ROOT/f"results/context_v2_{name}_20261030"
        tasks[name]=dict(status="pending",output_dir=str(out),log=str(folder/f"{name}.log"))
    atomic(status_file,dict(status="running",started_at=now(),order=NAMES,tasks=tasks))
    for name in NAMES:
        task=tasks[name];out=Path(task["output_dir"])
        args=[sys.executable,"-B","-u",str(ROOT/"scripts/evaluate_context.py"),"--configuration",name,
              "--role","evaluation","--output-dir",str(out)]
        if (out/"run.json").exists():args.append("--resume")
        task.update(status="running",started_at=now(),command=args)
        atomic(status_file,dict(status="running",updated_at=now(),order=NAMES,tasks=tasks))
        with Path(task["log"]).open("a") as log:
            completed=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        task.update(exit_code=completed.returncode,finished_at=now())
        if completed.returncode!=0:
            task["status"]="failed"
            atomic(status_file,dict(status="failed",updated_at=now(),order=NAMES,tasks=tasks))
            raise RuntimeError(f"Formal {name} failed; inspect {task['log']}, fix prerequisite before continuing")
        summary=json.loads((out/"summary.json").read_text())
        if not all(d["complete_manifest"] and d["inferred_count"]==256 and d["execution_errors"]==0 for d in summary.values()):
            task["status"]="incomplete"
            atomic(status_file,dict(status="incomplete",updated_at=now(),order=NAMES,tasks=tasks))
            raise RuntimeError("Formal run incomplete despite successful process exit")
        task.update(status="complete",summary=summary,run_metadata_sha256=sha(out/"run.json"))
        atomic(status_file,dict(status="running",updated_at=now(),order=NAMES,tasks=tasks))
        print(name,"completed 512/512",flush=True)
    atomic(status_file,dict(status="complete",finished_at=now(),order=NAMES,tasks=tasks,total_inferences=3072,source_freeze_sha256=sha(folder/"source_freeze.json")))
    print("All 12 formal groups complete, 3072 records",flush=True)


if __name__=="__main__":main()
