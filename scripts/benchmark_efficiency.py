"""Thin cost/performance adapter around frozen HF and project patches.

All hooks/profilers are confined to explicit probe phases. Timed natural/first/
fixed generation uses native generate and retains production patch overhead.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import Counter

from evaluate_baseline import atomic, append, sha, digest, now
from experiment_runtime import ROOT, PROTOCOL, NAMES, configs, scientific_binding, patch_for, gpu_lock, clean_model
from context_runtime import load_context_model, build_question
from verify_skip import prefill, decode, Calls
from PIL import Image
import torch

DEFAULT = ROOT / "evidence/day2_evening/cost_perf_v1"


def gpu_state():
    fields="name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit"
    result=subprocess.run(["nvidia-smi","--query-gpu="+fields,"--format=csv"],capture_output=True,text=True)
    power={}
    for directory in Path('/sys/class/power_supply').glob('*'):
        power[directory.name]={p.name:p.read_text().strip() for p in directory.iterdir() if p.name in {'online','status','type','capacity'}}
    return dict(at=now(),gpu=result.stdout.strip(),query_error=result.stderr.strip(),power_supplies=power,
                compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True).strip())


def all_methods(model):
    return [(n,m.forward,tuple(m._forward_pre_hooks),tuple(m._forward_hooks)) for n,m in model.named_modules()]


def formal(name, sample):
    return json.loads((ROOT/f"results/context_v2_{name}_20261030/{sample['dataset']}/samples/row_{sample['row_index']:05d}.json").read_text())


def prepare(processor, model, name, sample, protocol):
    saved=formal(name,sample);row=saved['sample'];image=ROOT/row['image_file']
    assert sha(image)==row['image_sha256']==sample['image_sha256']
    question=build_question(sample['dataset'],row,protocol)
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=question)])]
    prompt=processor.apply_chat_template(messages,add_generation_prompt=True)
    inputs=processor(text=prompt,images=[Image.open(image).convert('RGB')],return_tensors='pt').to('cuda')
    assert prompt==saved['prompt'] and inputs['input_ids'].tolist()==saved['input_ids']
    assert int((inputs['input_ids']==model.config.image_token_id).sum())==saved['image_placeholder_token_count']
    details={k:dict(shape=list(v.shape),dtype=str(v.dtype),device=str(v.device)) for k,v in inputs.items()}
    assert {k:v['shape'] for k,v in details.items()}=={k:v['shape'] for k,v in saved['input_shapes'].items()}
    assert {k:v['dtype'] for k,v in details.items()}=={k:v['dtype'] for k,v in saved['input_shapes'].items()}
    return inputs,saved,dict(prompt=prompt,input_ids=saved['input_ids'],inputs=details,formal_record_sha256=sha(ROOT/f"results/context_v2_{name}_20261030/{sample['dataset']}/samples/row_{sample['row_index']:05d}.json"))


def clear_calls(pruning):
    if hasattr(pruning,'calls'):pruning.calls.clear()


def timed(fn):
    torch.cuda.synchronize()
    start_allocated=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    ev0.record();start=time.perf_counter()
    result=fn()
    ev1.record();torch.cuda.synchronize();wall=time.perf_counter()-start
    record=dict(wall_ms=wall*1000,cuda_event_ms=ev0.elapsed_time(ev1),start_allocated_mib=start_allocated/2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                increment_allocated_mib=(torch.cuda.max_memory_allocated()-start_allocated)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    return result,record


def generation_block(model, inputs, saved, sample, pruning, phase, protocol, cfg, output):
    args=dict(protocol['generation'])
    if phase=='first':args['max_new_tokens']=1
    if phase=='fixed16':args.update(max_new_tokens=16,min_new_tokens=16)
    length=inputs['input_ids'].shape[1];records=[]
    for repeat in range(cfg['warmups']+cfg['repetitions']):
        clear_calls(pruning)
        result,measurement=timed(lambda:model.generate(**inputs,**args))
        ids=result[:,length:].tolist();g=len(ids[0]);del result
        if phase=='first':assert ids[0]==saved['new_token_ids'][0][:1]
        elif phase=='fixed16':assert g==16,(sample,g)
        else:assert ids==saved['new_token_ids'],(sample,'generation differs from formal quality')
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
        record=dict(sample=sample,phase=phase,repeat=repeat-cfg['warmups'],warmup=repeat<cfg['warmups'],new_token_ids=ids,G=g,
                    generation=args,ended_on_eos=bool(ids[0] and ids[0][-1] in eos),**measurement)
        append(output/'measurements.jsonl',record);records.append(record)
    return records


def probe(model,inputs,pruning,protocol,saved):
    captures={};shapes={};counts=Counter();handles=[];heads=[];attention=[]
    before=all_methods(model)
    modules={'vision':model.model.vision_model,'encoder':model.model.vision_model.encoder,'connector':model.model.connector}
    for label,module in modules.items():
        def pre(mod,args,kwargs,label=label):
            counts[label]+=1
            if label not in captures:captures[label]=dict(args=args,kwargs=kwargs)
        def post(mod,args,kwargs,out,label=label):
            if 'output' not in captures[label]:
                value=out.last_hidden_state if hasattr(out,'last_hidden_state') else out
                captures[label]['output']=value.detach().clone()
        handles += [module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post,with_kwargs=True)]
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d)):
            def watch(mod,args,name=name):
                counts[name]+=1
                if name not in shapes:shapes[name]=dict(input=list(args[0].shape),weight=list(mod.weight.shape),class_name=type(mod).__name__)
                if name=='lm_head':heads.append(list(args[0].shape))
            handles.append(module.register_forward_pre_hook(watch))
    def attn_pre(mod,args,kwargs):
        h=kwargs.get('hidden_states',args[0] if args else None);cache=kwargs.get('past_key_values')
        old=cache.get_seq_length(0) if cache is not None else 0
        attention.append(dict(q=h.shape[1],key_before=int(old),s=int(old+h.shape[1])))
    handles.append(model.model.text_model.layers[0].self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
    try:
        clear_calls(pruning);result=model.generate(**inputs,**protocol['generation'])
        assert result[:,inputs['input_ids'].shape[1]:].tolist()==saved['new_token_ids'];del result
    finally:
        for handle in handles:handle.remove()
    assert all_methods(model)==before
    g=saved['generated_length'];assert counts['vision']==counts['encoder']==counts['connector']==1
    assert len(heads)==g and all(x==[1,1,model.config.text_config.hidden_size] for x in heads)
    assert attention[0]['q']==inputs['input_ids'].shape[1] and all(a['q']==1 for a in attention[1:])
    assert [a['s'] for a in attention]==list(range(inputs['input_ids'].shape[1],inputs['input_ids'].shape[1]+g))
    return captures,dict(passed=True,counts=dict(counts),operator_shapes=shapes,lm_head_inputs=heads,attention_q_s=attention,all_hooks_removed=True)


def stage_block(model,inputs,saved,sample,pruning,protocol,cfg,output):
    captured,proof=probe(model,inputs,pruning,protocol,saved)
    atomic(output/f"probe_{sample['dataset']}.json",dict(sample=sample,**proof))
    for stage in ['vision','encoder','connector']:
        cap=captured[stage];module={'vision':model.model.vision_model,'encoder':model.model.vision_model.encoder,'connector':model.model.connector}[stage]
        for repeat in range(cfg['warmups']+cfg['repetitions']):
            clear_calls(pruning)
            result,m=timed(lambda:module(*cap['args'],**cap['kwargs']))
            value=result.last_hidden_state if hasattr(result,'last_hidden_state') else result
            assert torch.equal(value,cap['output']),(stage,'replay differs')
            del result,value
            append(output/'measurements.jsonl',dict(sample=sample,phase='stages',stage=stage,repeat=repeat-cfg['warmups'],warmup=repeat<cfg['warmups'],output_exact=True,
                   memory_scope='module replay with captured inputs/outputs resident; not whole-request memory',**m))
        # Encoder/vision replay restored pruning state at identical positions for Connector.
    del captured,cap,module;gc.collect()
    token=torch.tensor([formal('baseline',sample)['new_token_ids'][0][:1]],device='cuda',dtype=torch.long)
    for repeat in range(cfg['warmups']+cfg['repetitions']):
        clear_calls(pruning);pref=prefill(model,inputs);cache=pref.past_key_values;del pref
        assert cache.get_seq_length()==inputs['input_ids'].shape[1]
        result,m=timed(lambda:decode(model,cache,inputs,token))
        lengths=[result.past_key_values.get_seq_length(i) for i in range(30)]
        assert len(set(lengths))==1 and lengths[0]==inputs['input_ids'].shape[1]+1
        del result,cache
        append(output/'measurements.jsonl',dict(sample=sample,phase='stages',stage='decode',repeat=repeat-cfg['warmups'],warmup=repeat<cfg['warmups'],q=1,key_length=lengths[0],input_token=token.tolist(),fresh_cache=True,**m))


def skip_trace(model,inputs,sample,name,output):
    from torch.profiler import profile,ProfilerActivity,record_function
    calls=Calls(model,[8,16,24]);handles=[];contexts={};head=[]
    for number,layer in enumerate(model.model.text_model.layers,1):
        for attr in ['gate_proj','up_proj','down_proj']:
            label=f'L{number:02d}.mlp.{attr}';module=getattr(layer.mlp,attr)
            def pre(mod,args,label=label):ctx=record_function(label);ctx.__enter__();contexts[label]=ctx
            def post(mod,args,out,label=label):contexts.pop(label).__exit__(None,None,None)
            handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    handles.append(model.lm_head.register_forward_pre_hook(lambda mod,args:head.append(list(args[0].shape))))
    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
            calls.stage='prefill';pre=prefill(model,inputs);token=pre.logits[:,-1:].argmax(-1)
            calls.stage='decode';result=decode(model,pre.past_key_values,inputs,token);torch.cuda.synchronize()
        calls.verify([8,16,24] if name=='skip' else [])
        events=[e for e in prof.events() if str(e.device_type).endswith('CPU')]
        marker=Counter(e.name for e in events if e.name.startswith('L'))
        for number in range(1,31):
            for attr in ['gate_proj','up_proj','down_proj']:
                assert marker[f'L{number:02d}.mlp.{attr}']==(0 if name=='skip' and number in [8,16,24] else 2)
        trace=output/'skip_trace.json';prof.export_chrome_trace(str(trace))
        atomic(output/'skip_trace_proof.json',dict(passed=True,sample=sample,configuration=name,calls=dict(calls.counts),marker_counts=dict(marker),lm_head_inputs=head,
               q_prefill=inputs['input_ids'].shape[1],q_decode=1,s_decode=result.past_key_values.get_seq_length(),trace_sha256=sha(trace),scope='one prefill plus one cached decode; CPU module markers count calls, CUDA annotations excluded'))
        del result,pre
    finally:
        calls.close()
        for handle in handles:handle.remove()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--configuration',choices=NAMES,required=True)
    parser.add_argument('--phase',choices=['natural','anchor_start','anchor_end','first','fixed16','probe','stages','skip_trace','smoke'],required=True)
    parser.add_argument('--output-root',type=Path,default=DEFAULT)
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/cost_perf_v1.json').read_text());samples=json.loads((DEFAULT/'samples.json').read_text())
    output=args.output_root.resolve()/args.phase/args.configuration
    if ROOT/'evidence/day2_afternoon' in output.parents:raise ValueError('Historical outputs are protected')
    output.mkdir(parents=True,exist_ok=True)
    binding=dict(configuration=args.configuration,phase=args.phase,scientific=scientific_binding(),performance_config_sha256=sha(ROOT/'configs/cost_perf_v1.json'),samples_sha256=sha(DEFAULT/'samples.json'),source_sha256=sha(Path(__file__)))
    if (output/'complete.json').exists():
        assert json.loads((output/'binding.json').read_text())==binding;print('Already complete',output);return
    if (output/'binding.json').exists():raise ValueError('Partial attempt exists; preserve it and use a new output root')
    atomic(output/'binding.json',binding);(output/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    protocol=json.loads(PROTOCOL.read_text());selected=samples['natural'] if args.phase in ['natural','first'] else samples['diagnostic']
    if args.phase in ['smoke','skip_trace']:selected=selected[:1]
    try:
        with gpu_lock():
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            assert all(int(p.strip())==os.getpid() for p in pids.splitlines() if p.strip()),'External GPU compute process present; do not compete'
            atomic(output/'gpu_start.json',gpu_state())
            model,processor=load_context_model(output/'runtime',protocol);original=all_methods(model)
            cf,contract=configs()
            with torch.inference_mode(),patch_for(model,args.configuration,cf,contract,trace=False) as pruning:
                patched=all_methods(model)
                for sample in selected:
                    inputs,saved,alignment=prepare(processor,model,args.configuration,sample,protocol)
                    atomic(output/f"input_{sample['dataset']}_{sample['row_index']}.json",alignment)
                    if args.phase in ['probe','stages']:
                        if args.phase=='probe':
                            captured,proof=probe(model,inputs,pruning,protocol,saved);del captured
                            atomic(output/f"probe_{sample['dataset']}.json",dict(sample=sample,**proof))
                        else:stage_block(model,inputs,saved,sample,pruning,protocol,cfg,output)
                    elif args.phase=='skip_trace':skip_trace(model,inputs,sample,args.configuration,output)
                    elif args.phase=='smoke':
                        result=model.generate(**inputs,**protocol['generation']);assert result[:,inputs['input_ids'].shape[1]:].tolist()==saved['new_token_ids'];del result
                    else:generation_block(model,inputs,saved,sample,pruning,args.phase,protocol,cfg,output)
                    assert all_methods(model)==patched,'External probe hooks/methods leaked into timing'
                    del inputs,saved;gc.collect()
            clean_model(model);assert all_methods(model)==original
            atomic(output/'gpu_end.json',gpu_state())
        atomic(output/'complete.json',dict(passed=True,at=now(),samples=len(selected),command=sys.argv,binding_id=digest(binding),external_hooks_and_patch_restored=True))
        print('Completed',args.phase,args.configuration,len(selected),flush=True)
    except Exception as exc:
        atomic(output/'failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
