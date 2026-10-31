"""Relocated release checks using frozen verification and benchmark helpers.

Outputs must be outside this package. Historical strict entrypoints are intact.
"""
import argparse
from collections import Counter
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch as temporary_patch

import torch
import replay_release as replay
from replay_release import ROOT, local, read, binding_for
from evaluate_baseline import atomic, append, now, sha, digest
from evaluate_context import manifests_for
from context_runtime import load_context_model, build_question
from experiment_runtime import NAMES, patch_for, clean_model
from encoder_topk import EncoderTopKPatch
from verify_context import sample_run
from verify_encoder_topk import equal, toy_checks, signature, verify_path
import verify_skip
import benchmark_efficiency as bench
from release_profile import encoder_trace

SETTINGS = ROOT/'configs/release_performance_v3.json'
SAMPLES = ROOT/'configs/release_validation_samples_v3.json'


def data_ready(protocol, out):
    manifests = manifests_for(protocol)
    rows = [r for values in manifests.values() for r in values]
    assert len(rows) == 512
    paths = []
    for row in rows:
        image = local(ROOT/row['image_file'])
        assert sha(image) == row['image_sha256']
        paths.append(dict(path=str(image), sha256=row['image_sha256'], stable_id=row['stable_id']))
    atomic(out/'data_ready.json',dict(passed=True,at=now(),root=str(ROOT),counts={k:len(v) for k,v in manifests.items()},images=paths,
           scope='fixed manifests and packaged PNG bytes; no resampling or full-data download'))
    return manifests


def correctness(model, processor, manifests, cfgs, contract, protocol, out):
    report = dict(toys=toy_checks(),equivalence={},native={},skip={})
    methods, weights = bench.all_methods(model), signature(model)
    first = {k:v[0] for k,v in manifests.items()}
    for key,row in first.items():
        native, base = sample_run(model,processor,key,row,protocol,out/'native'/key)
        formal = local(ROOT/f"results/context_v2_baseline_20261030/{key}/samples/row_{row['row_index']:05d}.json")
        assert native['new_token_ids'] == read(formal)['new_token_ids']
        report['native'][key] = dict(stable_id=row['stable_id'],new_token_ids=native['new_token_ids'],input_ids=native['input_ids'],
            image=str(local(ROOT/row['image_file'])),formal=str(formal),formal_sha256=sha(formal))
        for name in NAMES[2:]:
            cfg = copy.deepcopy(cfgs[name]); cfg['keep_rates'] = [1.] * len(cfg['keep_rates'])
            with EncoderTopKPatch(model,cfg,contract) as pruning:
                got, seen = sample_run(model,processor,key,row,protocol,out/(name+'_100')/key)
                comparisons = {k:equal(base.values[k],seen.values[k]) for k in ['encoder','postnorm','connector','prefill_logits']}
                assert got['input_ids'] == native['input_ids'] and got['new_token_ids'] == native['new_token_ids']
                assert len(pruning.calls)==1 and pruning.calls[0]['bypass']
                assert pruning.calls[0]['score_calls']==pruning.calls[0]['scatter_calls']==0
                report['equivalence'][name+'/'+key] = dict(stable_id=row['stable_id'],effective_keep_rates=cfg['keep_rates'],
                    comparisons=comparisons,native_ids=native['new_token_ids'],patched_ids=got['new_token_ids'],generated_ids_equal=True,trace=copy.deepcopy(pruning.calls))
            clean_model(model)
            assert bench.all_methods(model)==methods and signature(model)==weights
            atomic(out/'correctness_partial.json',report)
        del base, seen; gc.collect()
    # This changes the diagnostic question builder only, exactly as verify_context does.
    # It never replaces environment guards, configs, comparison functions or expectations.
    with temporary_patch.object(verify_skip,'build_question',build_question):
        report['skip']['empty'] = verify_skip.empty_stage(model,processor,first,protocol,cfgs['skip'],out)
    generated = {}
    for phase in ['empty_original','empty_patched']:
        generated[phase] = {str(p.relative_to(out/phase)):read(p) for p in (out/phase).rglob('*.json') if 'new_token_ids' in read(p)}
    assert len(generated['empty_original'])==len(generated['empty_patched'])==2
    report['skip']['full_generated_records'] = generated
    assert bench.all_methods(model)==methods and signature(model)==weights
    report['cleanup'] = dict(all_named_module_forward_and_hook_sets_equal=True,weight_identity_pointer_version_equal=True,
        scope='normal exit checked; controlled exception patch exits below; no claim about arbitrary OOM/kill')
    for name in NAMES[1:]:
        try:
            with patch_for(model,name,cfgs,contract,trace=False):
                raise RuntimeError('controlled release cleanup probe')
        except RuntimeError as exc:
            assert str(exc)=='controlled release cleanup probe'
        clean_model(model)
        assert bench.all_methods(model)==methods and signature(model)==weights
    report['cleanup']['controlled_exception_patch_exits']=NAMES[1:]
    atomic(out/'correctness.json',dict(passed=True,at=now(),results=report))


def performance(model, processor, name, phase, cfgs, contract, protocol, out):
    cfg=read(SETTINGS); samples=read(SAMPLES)['diagnostic']; methods=bench.all_methods(model)
    with patch_for(model,name,cfgs,contract,trace=False) as pruning:
        patched=bench.all_methods(model)
        for sample in samples:
            inputs,saved,alignment=bench.prepare(processor,model,name,sample,protocol)
            image=local(ROOT/saved['sample']['image_file'])
            formal=local(ROOT/f"results/context_v2_{name}_20261030/{sample['dataset']}/samples/row_{sample['row_index']:05d}.json")
            atomic(out/f"input_{sample['dataset']}.json",dict(alignment=alignment,image=str(image),formal=str(formal),formal_sha256=sha(formal)))
            state_before=bench.gpu_state()
            assert bench.all_methods(model)==patched
            if phase=='stages':
                bench.stage_block(model,inputs,saved,sample,pruning,protocol,cfg,out)
            else:
                bench.generation_block(model,inputs,saved,sample,pruning,phase,protocol,cfg,out)
            append(out/'hardware.jsonl',dict(sample=sample,before=state_before,after=bench.gpu_state(),scope='brackets seven repeats; hardware sampling outside timed block'))
            assert bench.all_methods(model)==patched,'External hooks leaked into timing'
            del inputs,saved;gc.collect()
    clean_model(model);assert bench.all_methods(model)==methods


def profiler(model,processor,manifests,cfgs,contract,protocol,out):
    row=manifests['textvqa'][0]; sample=dict(dataset='textvqa',row_index=row['row_index'],stable_id=row['stable_id'],image_sha256=row['image_sha256'])
    methods=bench.all_methods(model)
    inputs,saved,alignment=bench.prepare(processor,model,'baseline',sample,protocol)
    atomic(out/'input.json',dict(alignment=alignment,image=str(local(ROOT/row['image_file']))))
    multi_dir=out/'multi';multi_dir.mkdir()
    multi=encoder_trace(model,inputs,row,'multi',cfgs,contract,multi_dir)
    assert [x['physical_k'] for x in multi['patch_trace'][0]['stages']]==[512,256,128]
    assert bench.all_methods(model)==methods
    for name in ['baseline','skip']:
        dest=out/name;dest.mkdir()
        with patch_for(model,name,cfgs,contract,trace=False):
            patched=bench.all_methods(model)
            bench.skip_trace(model,inputs,sample,name,dest)
            assert bench.all_methods(model)==patched
        clean_model(model);assert bench.all_methods(model)==methods
        proof=read(dest/'skip_trace_proof.json')
        proof['selected_dense_counts']={f'{stage}:layer_{i:02d}.{attr}':proof['calls'].get(f'{stage}:layer_{i:02d}.{attr}',0)
            for stage in ['prefill','decode'] for i in [8,16,24] for attr in ['self_attn','post_attention_layernorm','mlp']}
        atomic(dest/'skip_trace_proof.json',proof)
    atomic(out/'profiler.json',dict(passed=True,at=now(),sample=sample,multi_three_stages=True,baseline_and_skip_prefill_decode=True,
        all_named_module_hooks_restored=True,scope='separate diagnostic traces; nested CPU/device times overlap, no benchmark claim'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['data','correctness','performance','profiler'],required=True)
    p.add_argument('--configuration',choices=NAMES,default='baseline')
    p.add_argument('--phase',choices=['natural','fixed16','first','stages'],default='natural')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out==ROOT or ROOT in out.parents:p.error('Use an empty output directory outside the package')
    if out.exists() and any(out.iterdir()):p.error('Preserve prior attempts; output must be empty')
    out.mkdir(parents=True,exist_ok=True);started=now()
    try:
        binding,cfgs,contract,protocol=binding_for(args.configuration)
        files=['scripts/release_checks.py','scripts/release_profile.py','scripts/replay_release.py','scripts/verify_context.py',
            'scripts/verify_encoder_topk.py','scripts/verify_skip.py','scripts/benchmark_efficiency.py','scripts/profile_context.py',
            'configs/release_performance_v3.json','configs/release_validation_samples_v3.json']
        binding.update(kind='release-validation-v3',mode=args.mode,phase=args.phase,command=sys.argv,started_at=started,
            release_source_sha256={f:sha(local(ROOT/f)) for f in files},
            release_module_files={n:str(local(Path(sys.modules[n].__file__))) for n in ['release_profile','verify_context','verify_encoder_topk','verify_skip','benchmark_efficiency']},
            entry_file=str(Path(__file__).resolve()),entry_sha256=sha(Path(__file__)),shared_interpreter=True)
        atomic(out/'binding.json',binding)
        manifests=data_ready(protocol,out)
        if args.mode!='data':
            with (out.parent/'.release_gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
                assert all(int(v.strip())==os.getpid() for v in pids.splitlines() if v.strip()),'External GPU compute active'
                atomic(out/'gpu_start.json',bench.gpu_state())
                model,processor=load_context_model(out/'runtime',protocol);methods=bench.all_methods(model)
                try:
                    with torch.inference_mode():
                        if args.mode=='correctness':correctness(model,processor,manifests,cfgs,contract,protocol,out)
                        elif args.mode=='performance':performance(model,processor,args.configuration,args.phase,cfgs,contract,protocol,out)
                        else:profiler(model,processor,manifests,cfgs,contract,protocol,out)
                finally:
                    clean_model(model)
                    assert bench.all_methods(model)==methods,'Final forward/hook set differs'
                atomic(out/'gpu_end.json',bench.gpu_state())
        atomic(out/'complete.json',dict(passed=True,started_at=started,finished_at=now(),command=sys.argv,binding_id=digest(binding),root=str(ROOT),
            scope='independent unpack on same GPU/shared interpreter and HF cache; no full quality/performance rerun or cold cross-machine claim'))
        print('PASS',args.mode,args.configuration,args.phase,str(out),flush=True)
    except Exception as exc:
        atomic(out/'failure.json',dict(started_at=started,finished_at=now(),command=sys.argv,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
