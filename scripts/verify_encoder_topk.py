"""Day-2 morning diagnostics only: controlled toys, 100%, Early/Multi 50%.

Reuses fixed manifests, native HF loader/generate, frozen scoring and atomic
JSON helper. No formal 256-per-dataset Encoder evaluation is launched here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from day2_runtime import ROOT, EVIDENCE, proof_binding, sha
from evaluate_baseline import atomic, now
from run_original import load_original_model, run_sample, build_question
from encoder_topk import EncoderTopKPatch, attention_scores, select_tokens, restore_grid
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

NAMES = ["early", "middle", "late", "multi"]


def equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    report = dict(shape=list(a.shape), dtype=str(a.dtype), exact_equal=torch.equal(a, b),
                  max_abs_error=float((a.float()-b.float()).abs().max()), atol=0.0, rtol=0.0)
    assert report["exact_equal"] and report["max_abs_error"] == 0.0, report
    return report


class Observe:
    """Read-only hooks at real module entrances; first visual/prefill tensors."""
    def __init__(self, model):
        self.handles, self.values, self.events = [], {}, {}
        def before(module, name):
            def hook(mod, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states", kwargs.get("inputs_embeds"))
                event = dict(shape=list(x.shape)) if isinstance(x, torch.Tensor) else {}
                if kwargs.get("attention_mask") is not None:
                    event["mask_shape"] = list(kwargs["attention_mask"].shape)
                if name == "embeddings":
                    event["valid_patches_per_block"] = kwargs["patch_attention_mask"].flatten(1).sum(1).tolist()
                self.events.setdefault(name, []).append(event)
            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        def after(module, name, attr=None):
            def hook(mod, args, out):
                if name not in self.values:
                    value = getattr(out, attr) if attr else out
                    if name == "prefill_logits":
                        value = value[:, -1:, :]
                    self.values[name] = value.detach().cpu().clone()
            self.handles.append(module.register_forward_hook(hook))
        for i, layer in enumerate(model.model.vision_model.encoder.layers, 1):
            before(layer.self_attn, f"L{i:02d}.attention")
            before(layer.mlp, f"L{i:02d}.ffn")
        before(model.model.vision_model.embeddings, "embeddings")
        after(model.model.vision_model.encoder, "encoder", "last_hidden_state")
        before(model.model.vision_model.post_layernorm, "postnorm_in")
        after(model.model.vision_model.post_layernorm, "postnorm")
        before(model.model.connector.modality_projection.proj, "linear_in")
        after(model.model.connector, "connector")
        before(model.model.text_model, "llm")
        after(model.lm_head, "prefill_logits")

    def close(self):
        for h in self.handles:
            h.remove()


def toy_checks():
    # All inputs in this function are artificial CPU tensors, not model outputs.
    valid = torch.tensor([[True, True, False, True]])
    w = torch.tensor([[[[.25, .5, 0., .25], [.25, .5, 0., .25], [0., 0., 1., 0.], [.25, .5, 0., .25]]]]).expand(1,2,4,4)
    s = attention_scores(w, valid)
    assert torch.equal(s[:,[0,1,3]], torch.tensor([[.25,.5,.25]])) and torch.isneginf(s[0,2])
    # Known tie + invalid high-scoring key, with unequal per-block K.
    h = torch.arange(16,dtype=torch.float32).reshape(2,4,2)
    p = torch.tensor([[9,2,5,42],[6,3,0,9]])
    v = torch.tensor([[True,True,True,False],[False,True,False,False]])
    scores = torch.tensor([[1.,1.,1.,999.],[999.,1.,999.,999.]])
    selected,pos,mask,indices,ks = select_tokens(h,p,v,scores,.5)
    assert ks == [2,1] and indices == [[1,2],[1]] and pos.tolist() == [[2,5],[3,-1]]
    assert selected.shape == (2,2,2) and mask.tolist()==[[True,True],[True,False]]
    # Two-stage composition uses the current sequence, not the first ranking.
    x=torch.arange(12,dtype=torch.float32).reshape(1,6,2);orig=torch.arange(6)[None];valid6=torch.ones((1,6),dtype=torch.bool)
    x1,p1,v1,i1,k1=select_tokens(x,orig,valid6,torch.tensor([[0.,9.,3.,7.,1.,8.]]),.5)
    x2,p2,v2,i2,k2=select_tokens(x1,p1,v1,torch.tensor([[1.,5.,2.]]),.5)
    assert p1.tolist()==[[1,3,5]] and p2.tolist()==[[3,5]] and i2==[[1,2]]
    grid,occupied=restore_grid(x2,p2,v2,6)
    assert torch.equal(grid[0,p2[0]],x2[0]) and not torch.count_nonzero(grid[~occupied])
    # Nonzero LayerNorm bias illustrates why zero filling precedes no later norm.
    norm=torch.nn.LayerNorm(2)
    with torch.no_grad():norm.bias.copy_(torch.tensor([3.,-2.]))
    normalized=norm(x2);right,occ=restore_grid(normalized,p2,v2,6)
    wrong=norm(grid)
    assert not torch.count_nonzero(right[~occ]) and torch.count_nonzero(wrong[~occ])
    return dict(controlled_cpu_tensors_not_real_samples=True,head_query_axis=True,invalid_queries_excluded=True,
                invalid_keys_excluded=True,tie_selected_original_positions=pos.tolist(),effective_k=ks,physical_k=2,
                stage1_positions=p1.tolist(),stage2_positions=p2.tolist(),padding_not_scattered=True,
                postnorm_then_scatter_preserves_zero=True)


def sample_run(model, processor, key, sample, protocol, directory):
    observer = Observe(model)
    try:
        record = run_sample(model, processor, key, ROOT/sample["image_file"], build_question(key,sample,protocol),
                            sample=sample, output_dir=directory, protocol=protocol, trace_enabled=False, allow_empty=True)
    finally:
        observer.close()
    return record, observer


def signature(model):
    return {n:(id(p),p.data_ptr(),p._version,tuple(p.shape)) for n,p in model.named_parameters()}


def verify_path(patch_report, observations):
    assert len(patch_report)==1, "Only one visual prefill was expected with native use_cache"
    trace=patch_report[0]
    assert trace["installed"] and not trace["bypass"]
    assert trace["dropped_positions_exact_zero"] and trace["retained_scatter_exact"] and trace["scatter_calls"]==1
    assert trace["score_calls"]==len(trace["layers_1based"])
    stages={s["layer"]:s for s in trace["stages"]}
    for event in trace["layer_lengths"]:
        i=event["layer"]
        attn=observations.events[f"L{i:02d}.attention"]
        ffn=observations.events[f"L{i:02d}.ffn"]
        assert len(attn)==len(ffn)==1
        assert attn[0]["shape"][1]==event["attention_physical_length"]
        assert ffn[0]["shape"][1]==event["ffn_physical_length"]
        assert attn[0]["mask_shape"][-2:]==[event["attention_physical_length"]]*2
        if i in stages:
            stage=stages[i]
            assert stage["physical_k"]==max(stage["effective_k"])==ffn[0]["shape"][1]
            assert stage["next_mask_shape"][-2:]==[stage["physical_k"]]*2
            for b,chosen in enumerate(stage["selected_current_indices"]):
                before=stage["original_positions_before"][b]
                after=stage["original_positions_after"][b][:stage["effective_k"][b]]
                assert after==[before[j] for j in chosen] and after==sorted(after) and len(set(after))==len(after)
    assert observations.events["postnorm_in"][0]["shape"]==trace["encoder_output"]==trace["postnorm_output"]
    return True


def controlled_padding(model, config, contract):
    # Native learned Encoder/Connector, but synthetic already-embedded input.
    # No real image/question, no metric; records explicit physical padding cost.
    hidden=torch.arange(2*16*768,device="cuda",dtype=torch.float32).sin().reshape(2,16,768).to(torch.bfloat16)
    valid=torch.zeros((2,16),device="cuda",dtype=torch.bool);valid[0]=True;valid[1,[0,2,4,6,8]]=True
    mask=_prepare_4d_attention_mask(valid,hidden.dtype)
    observer=Observe(model)
    with EncoderTopKPatch(model,config,contract) as patch:
        try:
            out=model.model.vision_model.encoder(inputs_embeds=hidden,attention_mask=mask).last_hidden_state
            post=model.model.vision_model.post_layernorm(out)
            result=model.model.connector(post)
            verify_path(patch.calls,observer)
        finally:observer.close()
        return dict(controlled_embedded_tensor_not_image=True,original_valid=[16,5],
                    patch_trace=patch.calls,module_observations=observer.events,connector_shape=list(result.shape))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["toys","equivalence","early","multi"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve()
    if EVIDENCE not in output.parents:
        raise ValueError("Use a new day-2 verification directory")
    output.mkdir(parents=True,exist_ok=True)
    path=output/f"{args.stage}.json"
    if path.exists():raise ValueError("Do not overwrite completed verification reports")
    paths=[ROOT/f"configs/encoder_{name}.json" for name in NAMES]
    binding=proof_binding(paths)
    prereq={"equivalence":"toys","early":"equivalence","multi":"early"}.get(args.stage)
    if prereq:
        prior=json.loads((output/f"{prereq}.json").read_text());assert prior["passed"] and prior["binding"]==binding
    contract=json.loads((ROOT/"configs/encoder_topk_contract.json").read_text())
    configs={n:json.loads(p.read_text()) for n,p in zip(NAMES,paths)}
    for cfg in configs.values():
        assert cfg["contract_sha256"]==binding["contract_sha256"] and cfg["environment_id"]==binding["environment_id"]
        assert cfg["indices_0based"]==[i-1 for i in cfg["layers_1based"]]
    if args.stage=="toys":
        result=toy_checks()
    else:
        protocol=json.loads((ROOT/"configs/eval_protocol.json").read_text())
        model,processor=load_original_model(output/("runtime_"+args.stage),protocol)
        assert all(not hasattr(l,"_ffn_skip_original_forward") for l in model.model.text_model.layers)
        initial=signature(model);count=sum(p.numel() for p in model.parameters())
        samples={k:json.loads((ROOT/f"data/manifests/{stem}_validation_256.jsonl").read_text().splitlines()[0])
                 for k,stem in [("textvqa","textvqa"),("scienceqa","scienceqa_img")]}
        result={}
        with torch.inference_mode():
            if args.stage=="equivalence":
                for key,sample in samples.items():
                    baseline,base=sample_run(model,processor,key,sample,protocol,output/"native"/key)
                    checks={}
                    for name,cfg in configs.items():
                        effective=copy.deepcopy(cfg);effective["keep_rates"]=[1.]*len(cfg["keep_rates"])
                        with EncoderTopKPatch(model,effective,contract) as patch:
                            rec,obs=sample_run(model,processor,key,sample,protocol,output/(name+"_100")/key)
                            assert len(patch.calls)==1 and patch.calls[0]["installed"] and patch.calls[0]["bypass"]
                            assert patch.calls[0]["patched_layer_calls"]==len(cfg["layers_1based"])
                            assert patch.calls[0]["score_calls"]==patch.calls[0]["scatter_calls"]==0
                            comparisons={k:equal(base.values[k],obs.values[k]) for k in ["encoder","postnorm","connector","prefill_logits"]}
                            assert rec["input_ids"]==baseline["input_ids"] and rec["new_token_ids"]==baseline["new_token_ids"] and rec["raw_generated_text"]==baseline["raw_generated_text"]
                            checks[name]=dict(effective_keep_rates=effective["keep_rates"],comparisons=comparisons,
                                              generated_ids_equal=True,patch_assembly=patch.calls,module_observations=obs.events)
                        assert signature(model)==initial
                    result[key]=dict(stable_id=sample["stable_id"],input_length=baseline["input_length"],image_blocks=baseline["valid_image_blocks"],checks=checks)
            else:
                cfg=configs[args.stage]
                for key,sample in samples.items():
                    with EncoderTopKPatch(model,cfg,contract) as patch:
                        rec,obs=sample_run(model,processor,key,sample,protocol,output/args.stage/key)
                        verify_path(patch.calls,obs)
                        bridge=json.loads((EVIDENCE/f"environment/bridge/{key}/samples/row_{sample['row_index']:05d}.json").read_text())
                        assert rec["input_ids"]==bridge["input_ids"] and rec["image_placeholder_token_count"]==bridge["image_placeholder_token_count"]
                        assert obs.values["connector"].shape[0]*obs.values["connector"].shape[1]==rec["image_placeholder_token_count"]
                        assert obs.events["llm"][0]["shape"]==[1,rec["input_length"],model.config.text_config.hidden_size]
                        result[key]=dict(stable_id=sample["stable_id"],patch_trace=patch.calls,module_observations=obs.events,
                                         connector_output_shape=list(obs.values["connector"].shape),input_length=rec["input_length"],
                                         visual_token_count=rec["image_placeholder_token_count"],raw=rec["raw_generated_text"],score=rec["score"],generated_length=rec["generated_length"])
                    assert signature(model)==initial
                result["controlled_padding"]=controlled_padding(model,cfg,contract)
            assert signature(model)==initial and sum(p.numel() for p in model.parameters())==count
        result["parameter_count_unchanged"]=count
        result["llm_unmodified_no_skip"]=True
    atomic(path,dict(stage=args.stage,passed=True,binding=binding,recorded_at=now(),command=sys.argv,results=result))
    print(json.dumps(dict(stage=args.stage,passed=True,report=str(path)),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
