"""New predictions with frozen context-v2 helpers; historical replay is separate.

The artifact root supplies the scientific code, configs, manifests and evidence.
No monkeypatch of ROOT, no disabled assertions and no replacement model/scorer.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--configuration', choices=['baseline','early','middle','late','multi','skip'], required=True)
    p.add_argument('--limit', type=int, default=256, help='First 1..256 rows of each fixed manifest; less than 256 is partial')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--history-compare', action='store_true', help='Save diagnostic differences, never require matching predictions')
    a = p.parse_args()
    if not 1 <= a.limit <= 256:
        p.error('limit must be 1..256')
    root = a.artifact_root.resolve(strict=True)
    out = a.output_dir.resolve()
    if root == out or root in out.parents:
        p.error('Output must be outside the scientific artifact root')
    if out.exists() and any(out.iterdir()):
        p.error('Use a new empty directory; preserve all previous attempts')
    if sys.flags.optimize:
        p.error('Assertions must remain enabled')
    sys.path.insert(0, str(root / 'scripts'))
    from replay_release import binding_for, local
    from context_runtime import load_context_model, build_question
    from evaluate_context import manifests_for
    from evaluate_baseline import atomic, now, sha, digest, record_path, summarize
    from experiment_runtime import ROOT, patch_for, clean_model, method_signature
    from run_original import run_sample
    import torch
    assert ROOT == root, (ROOT, root)
    out.mkdir(parents=True, exist_ok=True)
    binding, cfgs, contract, protocol = binding_for(a.configuration)
    binding.update(kind='new-prediction-evaluation-v1', entry_file=str(Path(__file__).resolve()),
                   entry_sha256=sha(Path(__file__)), argv=sys.argv,
                   history_comparison_required=False,
                   validation_scope='frozen science/dependencies, real current Python/GPU binding; no claim of cold install')
    binding_id = digest(binding)
    atomic(out / 'binding.json', binding)
    (out / 'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifests = manifests_for(protocol)
    selected = [(k,r) for k,rows in manifests.items() for r in rows[:a.limit]]
    comparisons = []
    started = now()

    def aggregate():
        result = summarize(out, manifests, binding_id, a.limit, 'new-prediction-evaluation')
        # Preserve the original aggregation fields. Add an explicit requested
        # denominator and partial flag; an exception never becomes a complete score.
        for key, s in result.items():
            s.update(partial=a.limit < 256,
                     requested_mean_score=s['partial_mean_score'] if s['complete_requested'] else None,
                     requested_score_denominator=s['requested_target'],
                     historical_quality_table_addition=False)
            atomic(out / key / 'summary.json', s)
        atomic(out / 'summary.json', result)
        return result

    for key in manifests:
        (out/key/'samples').mkdir(parents=True, exist_ok=True)
    try:
        # Same per-parent lock convention as the release checks; also reject any
        # external compute process, including a run using another output parent.
        with (out.parent/'.release_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
            assert all(int(x.strip()) == os.getpid() for x in pids.splitlines() if x.strip()), 'External GPU compute active'
            model, processor = load_context_model(out/'runtime', protocol)
            methods = method_signature(model)
            try:
                with torch.inference_mode(), patch_for(model,a.configuration,cfgs,contract,trace=False) as patch:
                    for key,row in selected:
                        tick = time.monotonic()
                        try:
                            image = local(root/row['image_file'])
                            assert sha(image) == row['image_sha256']
                            question = build_question(key,row,protocol)
                            if hasattr(patch,'calls'):
                                patch.calls.clear()
                            result = run_sample(model,processor,key,image,question,sample=row,
                                                output_dir=out/'runtime',trace_enabled=False,
                                                persist=False,allow_empty=True,protocol=protocol)
                            assert result['question_for_model'] == question
                            result.update(status='success',inference_status='ok',stable_id=row['stable_id'],
                                          row_index=row['row_index'],configuration=a.configuration,binding_id=binding_id,
                                          protocol_sha256=binding['protocol_sha256'],
                                          scheduling_sample_wall_seconds=time.monotonic()-tick,
                                          timing_scope='scheduling only, includes Processor/scoring; not benchmark latency')
                            if hasattr(patch,'calls'):
                                result['encoder_computation'] = copy.deepcopy(patch.calls)
                            atomic(record_path(out,key,row),result)
                            if a.history_compare:
                                reference = root/f'results/context_v2_{a.configuration}_20261030/{key}/samples/row_{row["row_index"]:05d}.json'
                                old = json.loads(reference.read_text())
                                fields = ['prompt','input_ids','input_shapes','new_token_ids','raw_generated_text','score','parse_status','generated_length','input_length']
                                comparisons.append(dict(dataset=key,stable_id=row['stable_id'],reference_sha256=sha(reference),
                                                        equal={f:result[f] == old[f] for f in fields}))
                            print(key,row['stable_id'],result['score'],result['parse_status'],flush=True)
                        except Exception as exc:
                            # Keep the requested denominator and explicit error, never
                            # silently drop a failed item or substitute historical output.
                            error=dict(status='error',inference_status='exception',sample=row,
                                       stable_id=row['stable_id'],row_index=row['row_index'],configuration=a.configuration,
                                       binding_id=binding_id,score=None,error_type=type(exc).__name__,error=str(exc),
                                       traceback=traceback.format_exc(),recorded_at=now())
                            atomic(record_path(out,key,row),error)
                            raise
            finally:
                clean_model(model)
                assert method_signature(model) == methods, 'Patch/hook leaked'
        summary = aggregate()
        atomic(out/'history_comparison.json',dict(enabled=a.history_compare,diagnostic_only=True,rows=comparisons))
        assert all(s['complete_requested'] and not s['execution_errors'] for s in summary.values())
        atomic(out/'complete.json',dict(passed=True,started_at=started,finished_at=now(),count=len(selected),
                                      partial=a.limit<256,model_loads=1,binding_id=binding_id,summary=summary))
    except Exception:
        summary = aggregate()
        atomic(out/'failure.json',dict(passed=False,started_at=started,finished_at=now(),traceback=traceback.format_exc(),summary=summary))
        raise


if __name__ == '__main__':
    main()
