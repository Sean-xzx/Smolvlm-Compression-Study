"""Relocated release replay; native HF and frozen patches remain unchanged.

A new environment binding replaces no historical proof: original quality entry
and assert_environment are retained. Only this entry accepts relocation.
"""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from context_runtime import load_context_model, build_question
from evaluate_baseline import atomic, now, sha, digest
from evaluate_context import manifests_for
from experiment_runtime import ROOT, NAMES, config_path, patch_for, clean_model, method_signature
from run_original import run_sample
import torch
import encoder_topk
import ffn_skip


def read(path):
    return json.loads(path.read_text())


def local(path):
    resolved = path.resolve()
    if ROOT not in resolved.parents:
        raise ValueError('Project input/module outside unpacked root: '+str(resolved))
    return resolved


def binding_for(name):
    historical = read(ROOT/'results/context_v2_baseline_20261030/run.json')['binding']['common']['scientific']
    for field in ['configurations_sha256', 'manifests_sha256', 'core_code_sha256']:
        for rel, expected in historical[field].items():
            assert sha(local(ROOT/rel)) == expected, rel
    cfgs = {n:read(config_path(n)) for n in NAMES}
    contract = read(ROOT/'configs/context_v2/encoder_topk_contract.json')
    protocol = read(ROOT/'configs/eval_protocol_context_v2.json')
    assert sha(ROOT/'configs/eval_protocol_context_v2.json') == historical['protocol_sha256']
    assert protocol['status'] == 'frozen' and contract['base_protocol_sha256'] == historical['protocol_sha256']
    for n,cfg in cfgs.items():
        assert cfg['status'] == 'frozen'
        assert cfg.get('base_protocol_sha256',cfg.get('baseline_protocol_sha256')) == historical['protocol_sha256']
        if n not in ['baseline','skip']:
            assert cfg['contract_sha256'] == sha(ROOT/'configs/context_v2/encoder_topk_contract.json')
    sources = {}
    expected_native = {Path(p).name:h for p,h in historical['environment']['sources_sha256'].items()}
    for module in ['idefics3.modeling_idefics3','idefics3.processing_idefics3','idefics3.image_processing_idefics3','llama.modeling_llama']:
        p = Path(importlib.util.find_spec('transformers.models.'+module).origin)
        assert sha(p) == expected_native[p.name], p
        sources[str(p)] = sha(p)
    packages = {n:importlib.metadata.version(n) for n in historical['environment']['dependencies']}
    assert packages == historical['environment']['dependencies'], 'Dependency version differs from measured reference'
    modules = {n:str(local(Path(sys.modules[n].__file__))) for n in ['context_runtime','run_original','check_scores','evaluate_baseline','evaluate_context','experiment_runtime','day2_runtime','encoder_topk','ffn_skip']}
    environment = dict(python=sys.version,executable=sys.executable,resolved_executable=str(Path(sys.executable).resolve()),
                       python_binary_sha256=sha(Path(sys.executable).resolve()),dependencies=packages,native_sources_sha256=sources,
                       gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'],text=True).strip())
    binding = dict(kind='release-replay-v1',configuration=name,root=str(ROOT),entry_file=str(Path(__file__).resolve()),
                   entry_sha256=sha(Path(__file__)),cwd=os.getcwd(),environment=environment,current_environment_id=digest(environment),
                   historical_environment_id_reference=historical['environment_id'],historical_proof_claim='reference only, not a new historical passed proof',
                   scientific=historical,local_module_files=modules,shared_HF_HOME=os.environ.get('HF_HOME'),
                   protocol_sha256=sha(ROOT/'configs/eval_protocol_context_v2.json'))
    return binding,cfgs,contract,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',choices=NAMES,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=1,help='First 1..256 rows from each fixed manifest; default 1; full 256 capability not tested this release')
    args=parser.parse_args()
    if not 1<=args.limit<=256:parser.error('limit must be 1..256')
    out=args.output_dir.resolve()
    if out==ROOT or ROOT in out.parents:parser.error('Use a new output directory outside the unpacked project; evidence stays read-only')
    if out.exists() and any(out.iterdir()):parser.error('Preserve existing attempt; new empty output required')
    out.mkdir(parents=True,exist_ok=True)
    binding,cfgs,contract,protocol=binding_for(args.configuration)
    atomic(out/'binding.json',binding)
    (out/'replay_source.py').write_bytes(Path(__file__).read_bytes())
    manifests=manifests_for(protocol)
    selected=[(key,row) for key,rows in manifests.items() for row in rows[:args.limit]]
    with (out.parent/'.release_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        assert all(int(p.strip())==os.getpid() for p in pids.splitlines() if p.strip()),'External GPU compute active'
        model,processor=load_context_model(out/'runtime',protocol)
        methods=method_signature(model);checks=[]
        try:
            with torch.inference_mode(),patch_for(model,args.configuration,cfgs,contract,trace=False) as patch:
                for key,row in selected:
                    image=local(ROOT/row['image_file']);assert sha(image)==row['image_sha256']
                    formal_path=local(ROOT/f"results/context_v2_{args.configuration}_20261030/{key}/samples/row_{row['row_index']:05d}.json")
                    saved=read(formal_path);counts=Counter();shapes={};handles=[]
                    def watch(label):
                        def hook(module,args,kwargs=None):
                            counts[label]+=1
                            x=args[0] if args else (kwargs or {}).get('hidden_states')
                            if x is not None:shapes.setdefault(label,[]).append(list(x.shape))
                        return hook
                    if args.configuration=='skip':
                        for n in cfgs['skip']['layers_1based']:
                            layer=model.model.text_model.layers[n-1]
                            handles.extend([layer.mlp.register_forward_pre_hook(watch(f'L{n}.mlp')),
                                            layer.self_attn.register_forward_pre_hook(watch(f'L{n}.attn'),with_kwargs=True)])
                    elif args.configuration!='baseline':
                        for n in cfgs[args.configuration]['layers_1based']:
                            handles.append(model.model.vision_model.encoder.layers[n-1].mlp.register_forward_pre_hook(watch(f'L{n}.ffn')))
                    if hasattr(patch,'calls'):patch.calls.clear()
                    try:
                        result=run_sample(model,processor,key,image,build_question(key,row,protocol),sample=row,
                                          output_dir=out/'runtime',trace_enabled=False,persist=False,allow_empty=True,protocol=protocol)
                    finally:
                        for h in handles:h.remove()
                    fields=['prompt','input_ids','input_shapes','new_token_ids','raw_generated_text','score','parse_status','generated_length','input_length']
                    comparison={f:result[f]==saved[f] for f in fields};assert all(comparison.values()),comparison
                    if args.configuration=='skip':
                        # Attention uses keyword hidden_states, so count hook must not depend on positional args.
                        for n in cfgs['skip']['layers_1based']:assert counts[f'L{n}.mlp']==0 and counts[f'L{n}.attn']==result['generated_length']
                    elif args.configuration!='baseline':
                        assert len(patch.calls)==1
                        stages=patch.calls[0]['stages'];assert len(stages)==len(cfgs[args.configuration]['layers_1based'])
                        for stage in stages:
                            expected=[result['valid_image_blocks'],stage['physical_k'],768]
                            assert shapes[f"L{stage['layer']}.ffn"]==[expected]
                    result.update(configuration=args.configuration,stable_id=row['stable_id'],input_absolute_path=str(image),formal_absolute_path=str(formal_path),
                                  formal_sha256=sha(formal_path),binding_id=digest(binding),comparison=comparison,hook_counts=dict(counts),hook_shapes=shapes,
                                  encoder_calls=patch.calls if hasattr(patch,'calls') else None,scope='release replay only; not quality-table addition or latency benchmark')
                    atomic(out/key/f"row_{row['row_index']:05d}.json",result)
                    checks.append(dict(dataset=key,row_index=row['row_index'],all_fields_equal=True,G=result['generated_length'],hook_counts=dict(counts),hook_shapes=shapes))
        finally:
            clean_model(model)
            assert method_signature(model)==methods,'Patch/hook leaked'
        atomic(out/'complete.json',dict(passed=True,at=now(),command=sys.argv,root=str(ROOT),binding_id=digest(binding),checks=checks,count=len(checks),
                                      scope='same-machine unpacked code/config/inputs; external interpreter/HF cache; not cold cross-machine validation'))


if __name__=='__main__':
    try:main()
    except Exception:
        # Preserve stdout/stderr and any atomic rows; never turn a partial run into passed.
        traceback.print_exc();raise
