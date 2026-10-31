"""Short diagnostic profiler with per-layer shape/matmul evidence; no benchmark."""
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch as temporary_patch

from evaluate_baseline import atomic,now,sha
from experiment_runtime import ROOT,BASE,PROTOCOL,MANIFESTS,NAMES,scientific_binding,configs,patch_for,gpu_lock,clean_model
from context_runtime import load_context_model,build_question
from verify_skip import inputs_for as original_inputs
from verify_encoder_topk import Observe
import verify_skip
import encoder_topk
import torch
from torch.profiler import profile,ProfilerActivity,record_function


def main():
    output=BASE/"b4";output.mkdir(parents=True,exist_ok=True)
    if (output/"report.json").exists():raise ValueError("Completed profiler exists")
    assert json.loads((BASE/"b3/verification.json").read_text())["passed"]
    protocol=json.loads(PROTOCOL.read_text());cfgs,contract=configs();binding=scientific_binding()
    binding["profiler_code_sha256"]=sha(Path(__file__))
    row=json.loads(MANIFESTS[0].read_text().splitlines()[0]);key=row["dataset_key"]
    reports={}
    with gpu_lock():
        model,processor=load_context_model(output/"runtime",protocol)
        with temporary_patch.object(verify_skip,"build_question",build_question):
            inputs=original_inputs(processor,key,row,protocol)
        with torch.inference_mode():
            for name in ["baseline",*NAMES[2:]]:
                with patch_for(model,name,cfgs,contract,trace=False) as pruning:
                    # One unprofiled forward initializes kernels; not a timed benchmark warmup.
                    model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                    torch.cuda.synchronize()
                    if pruning is not None:pruning.calls.clear()
                    handles=[];contexts={};observations={}
                    def mark(module,label):
                        def before(mod,args):
                            observations[label]=list(args[0].shape) if args else None
                            ctx=record_function(label);ctx.__enter__();contexts[label]=ctx
                        def after(mod,args,out):contexts.pop(label).__exit__(None,None,None)
                        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
                    for number,layer in enumerate(model.model.vision_model.encoder.layers,1):
                        mark(layer,f"Vision.L{number:02d}")
                        for attr in ["q_proj","k_proj","v_proj","out_proj"]:
                            mark(getattr(layer.self_attn,attr),f"Vision.L{number:02d}.attention.{attr}")
                        for attr in ["fc1","fc2"]:
                            mark(getattr(layer.mlp,attr),f"Vision.L{number:02d}.mlp.{attr}")
                    observer=Observe(model)
                    def wrap(function,label):
                        def wrapped(*args,**kwargs):
                            if pruning is not None and label!="scatter":
                                index=pruning.report["score_calls"]-(1 if label=="select_sort_gather" else 0)
                                stage=list(pruning.rates)[index]
                                marker=f"Vision.L{stage:02d}.{label}"
                            else:marker="Vision."+label
                            with record_function(marker):return function(*args,**kwargs)
                        return wrapped
                    try:
                        with ExitStack() as stack:
                            for function,label in [("attention_scores","score_fp32"),("select_tokens","select_sort_gather"),("restore_grid","scatter")]:
                                stack.enter_context(temporary_patch.object(encoder_topk,function,wrap(getattr(encoder_topk,function),label)))
                            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
                                with record_function(name+"_single_prefill"):
                                    pred=model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
                                torch.cuda.synchronize()
                        trace_path=output/f"{name}_trace.json";prof.export_chrome_trace(str(trace_path))
                    finally:
                        observer.close()
                        for handle in handles:handle.remove()
                    events=list(prof.events())
                    cpu=[e for e in events if str(e.device_type).endswith("CPU")]
                    markers=[e for e in cpu if e.name.startswith("Vision.")]
                    counts=Counter(e.name for e in markers)
                    assert all(c==1 for c in counts.values()),"CPU markers counted once; GPU annotations are not module calls"
                    matrices=[]
                    for e in cpu:
                        if e.name not in {"aten::mm","aten::addmm","aten::bmm"}:continue
                        parent=e.cpu_parent
                        while parent and not parent.name.startswith("Vision."):parent=parent.cpu_parent
                        if parent:
                            matrices.append(dict(marker=parent.name,operator=e.name,input_shapes=e.input_shapes,
                                                 cpu_us=e.cpu_time_total,cuda_us=e.device_time_total))
                    for number in range(1,13):
                        for suffix in ["attention.q_proj","attention.k_proj","attention.v_proj","attention.out_proj","mlp.fc1","mlp.fc2"]:
                            label=f"Vision.L{number:02d}."+suffix
                            assert counts[label]==1
                            assert len([m for m in matrices if m["marker"]==label and m["operator"] in {"aten::mm","aten::addmm"}])==1,label
                    layer_rows=[]
                    valid=observer.events["embeddings"][0]["valid_patches_per_block"]
                    stages={s["layer"]:s for s in pruning.calls[0]["stages"]} if pruning else {}
                    for number in range(1,13):
                        attn=observer.events[f"L{number:02d}.attention"][0]
                        ffn=observer.events[f"L{number:02d}.ffn"][0]
                        before=valid[:]
                        if number in stages:valid=stages[number]["effective_k"]
                        na,nf=attn["shape"][1],ffn["shape"][1];b=attn["shape"][0]
                        assert observations[f"Vision.L{number:02d}.attention.q_proj"]==[b,na,768]
                        assert observations[f"Vision.L{number:02d}.mlp.fc1"]==[b,nf,768]
                        layer_rows.append(dict(layer=number,B=b,Nattn=na,Nffn=nf,valid_before=before,valid_after=valid[:],
                                               padding_before=b*na-sum(before),padding_after=b*nf-sum(valid),
                                               hidden=model.config.vision_config.hidden_size,intermediate=model.config.vision_config.intermediate_size))
                    if pruning:
                        assert len(stages)==(3 if name=="multi" else 1)
                        for number in stages:
                            assert counts[f"Vision.L{number:02d}.score_fp32"]==counts[f"Vision.L{number:02d}.select_sort_gather"]==1
                        assert counts["Vision.scatter"]==1
                    cuda_events=sum(str(e.device_type).endswith("CUDA") for e in events)
                    assert cuda_events>0,"No CUDA events captured; do not invent GPU timing"
                    reports[name]=dict(stable_id=row["stable_id"],input_length=inputs["input_ids"].shape[1],
                        visual_positions=int((inputs["input_ids"]==model.config.image_token_id).sum()),layer_lengths=layer_rows,
                        module_input_shapes=observations,marker_counts=dict(counts),matrix_operators=matrices,
                        marker_times=[dict(name=e.name,cpu_us=e.cpu_time_total,cuda_us=e.device_time_total) for e in markers],
                        cuda_events=cuda_events,patch_trace=copy.deepcopy(pruning.calls) if pruning else None,
                        trace_file=str(trace_path.relative_to(ROOT)),trace_sha256=sha(trace_path),
                        scope="one diagnostic prefilling forward; record_shapes/hooks/CPU synchronization retained; nested times overlap; not a latency benchmark")
                    atomic(output/f"{name}.json",reports[name]);del pred
                clean_model(model)
        assert binding["environment_id"]==scientific_binding()["environment_id"]
    atomic(output/"report.json",dict(passed=True,recorded_at=now(),binding=binding,reports=reports,
        conclusion="Native matmul shapes match pre-hooks for baseline and all four Encoder configurations; Multi covers all 3 score stages",
        limitation="No speed or memory savings claim; Connector and LLM visual positions remain unshortened. Formal latency/memory and complete FLOPs accounting remain later work."))
    print("B4 passed",flush=True)


if __name__=="__main__":main()
