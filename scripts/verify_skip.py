"""Real GPU checks for the instance-only Skip patch; no formal evaluation.

Each stage writes its own report/log. Numeric checks require exact equality.
Prefill creates a fresh native cache per path; decode never shares that cache.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import sys

from evaluate_baseline import ROOT, sha, atomic, now
from run_original import load_original_model, build_question, run_sample
from ffn_skip import SkipPatch, skip_forward
import torch
from PIL import Image


def comparison(a, b):
    result = dict(shape=list(a.shape), dtype=str(a.dtype), exact_equal=torch.equal(a, b),
                  max_abs_error=float((a.float()-b.float()).abs().max().item()), atol=0.0, rtol=0.0)
    assert result["exact_equal"] and result["max_abs_error"] == 0.0, result
    return result


def inputs_for(processor, key, sample, protocol):
    image = Image.open(ROOT / sample["image_file"]).convert("RGB")
    question = build_question(key, sample, protocol)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor(text=processor.apply_chat_template(messages, add_generation_prompt=True),
                     images=[image], return_tensors="pt").to("cuda")


def prefill(model, inputs):
    return model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)


def decode(model, cache, inputs, tokens):
    n = inputs["input_ids"].shape[1]
    assert cache.get_seq_length() == n
    mask = torch.cat([inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))], dim=1)
    return model(input_ids=tokens, attention_mask=mask, past_key_values=cache,
                 position_ids=torch.tensor([[n]], device="cuda"),
                 cache_position=torch.tensor([n], device="cuda"),
                 use_cache=True, logits_to_keep=1, return_dict=True)


class Calls:
    """Temporary hooks: counts / input shapes / actual attention residual equality."""
    def __init__(self, model, selected, check_residual=False):
        self.handles, self.counts, self.shapes, self.residual_checks = [], Counter(), {}, []
        self.stage, self.saved_input, self.attn_output = "unset", {}, {}
        self.selected = selected
        def watch(module, name):
            def pre(mod, args, kwargs):
                key = self.stage + ":" + name
                self.counts[key] += 1
                x = args[0] if args else kwargs.get("hidden_states")
                if x is not None:
                    self.shapes.setdefault(key, []).append(list(x.shape))
            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        for i, layer in enumerate(model.model.text_model.layers, 1):
            for attr in ["self_attn", "input_layernorm", "post_attention_layernorm", "mlp"]:
                watch(getattr(layer, attr), f"layer_{i:02d}.{attr}")
            if i in selected and check_residual:
                def before(mod, args, kwargs, i=i):
                    self.saved_input[i] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
                def attn(mod, args, out, i=i):
                    self.attn_output[i] = out[0].detach().clone()
                def after(mod, args, out, i=i):
                    expected = self.saved_input.pop(i) + self.attn_output.pop(i)
                    self.residual_checks.append(dict(stage=self.stage, layer=i, **comparison(expected, out)))
                self.handles += [layer.register_forward_pre_hook(before, with_kwargs=True),
                                 layer.self_attn.register_forward_hook(attn), layer.register_forward_hook(after)]
        watch(model.model.text_model.norm, "final_norm")

    def verify(self, skipped):
        for stage in ["prefill", "decode"]:
            for i in range(1, 31):
                for attr in ["self_attn", "input_layernorm", "post_attention_layernorm", "mlp"]:
                    expected = 0 if i in skipped and attr in ["post_attention_layernorm", "mlp"] else 1
                    actual = self.counts[stage + f":layer_{i:02d}.{attr}"]
                    assert actual == expected, (stage, i, attr, actual, expected)
            assert self.counts[stage + ":final_norm"] == 1

    def close(self):
        for h in self.handles:
            h.remove()


def empty_stage(model, processor, samples, protocol, config, output):
    results = {}
    for key, sample in samples.items():
        inputs = inputs_for(processor, key, sample, protocol)
        a = prefill(model, inputs)
        token = a.logits[:, -1:].argmax(-1)
        cache_a = a.past_key_values
        b = None
        base_logits = a.logits.detach().clone()
        da = decode(model, cache_a, inputs, token)
        da_logits = da.logits.detach().clone()
        base_gen = run_sample(model, processor, key, ROOT/sample["image_file"], build_question(key, sample, protocol), sample=sample,
                              output_dir=output/"empty_original", trace_enabled=False, protocol=protocol, allow_empty=True)
        with SkipPatch(model, [], config["llama_source_sha256"]):
            b = prefill(model, inputs)
            assert b.past_key_values is not cache_a
            num = dict(prefill=comparison(base_logits, b.logits))
            cache_b = b.past_key_values
            assert all(x == inputs["input_ids"].shape[1] for x in [cache_b.get_seq_length(i) for i in range(30)])
            db = decode(model, cache_b, inputs, token)
            num["decode"] = comparison(da_logits, db.logits)
            gen = run_sample(model, processor, key, ROOT/sample["image_file"], build_question(key, sample, protocol), sample=sample,
                             output_dir=output/"empty_patched", trace_enabled=False, protocol=protocol, allow_empty=True)
        assert gen["new_token_ids"] == base_gen["new_token_ids"] and gen["raw_generated_text"] == base_gen["raw_generated_text"]
        results[key] = dict(stable_id=sample["stable_id"], numeric=num, decode_input_ids=token.tolist(),
                            independent_cache_objects=True, cache_lengths_a=[cache_a.get_seq_length(i) for i in range(30)],
                            cache_lengths_b=[cache_b.get_seq_length(i) for i in range(30)],
                            generated_ids_equal=True, original_output=base_gen["raw_generated_text"],
                            patched_output=gen["raw_generated_text"], input_length=gen["input_length"], generated_length=gen["generated_length"])
        del a, b, da, db, cache_a, cache_b
    return results


def enabled_stage(model, processor, samples, protocol, config, output):
    results = {}
    selected = config["layers_1based"]
    for key, sample in samples.items():
        inputs = inputs_for(processor, key, sample, protocol)
        baseline = Calls(model, selected)
        baseline.stage = "prefill"
        original = prefill(model, inputs)
        token = original.logits[:, -1:].argmax(-1)
        baseline.stage = "decode"
        decode(model, original.past_key_values, inputs, token)
        baseline.verify([])
        baseline.close()
        with SkipPatch(model, selected, config["llama_source_sha256"]):
            calls = Calls(model, selected, check_residual=True)
            try:
                calls.stage = "prefill"
                skipped = prefill(model, inputs)
                assert skipped.past_key_values is not original.past_key_values
                calls.stage = "decode"
                decoded = decode(model, skipped.past_key_values, inputs, token)
                calls.verify(selected)
                lengths = [decoded.past_key_values.get_seq_length(i) for i in range(30)]
                assert all(n == inputs["input_ids"].shape[1]+1 for n in lengths)
            finally:
                calls.close()
            # Separate real greedy generation; counts establish actual G-1 decode calls.
            generation_calls = Calls(model, selected)
            generation_calls.stage = "generate"
            try:
                generated = run_sample(model, processor, key, ROOT/sample["image_file"], build_question(key, sample, protocol), sample=sample,
                                       output_dir=output/"enabled_generation", trace_enabled=False, protocol=protocol, allow_empty=True)
            finally:
                generation_calls.close()
            g = generated["generated_length"]
            for i in range(1,31):
                assert generation_calls.counts[f"generate:layer_{i:02d}.mlp"] == (0 if i in selected else g)
            assert generation_calls.counts["generate:final_norm"] == g
        results[key] = dict(stable_id=sample["stable_id"], baseline_counts=dict(baseline.counts), skip_counts=dict(calls.counts),
                            skip_input_shapes=calls.shapes, residual_checks=calls.residual_checks, cache_lengths=lengths,
                            generation_calls=dict(generation_calls.counts), generation_input_shapes=generation_calls.shapes,
                            generated_length=g, input_length=generated["input_length"], raw_generated_text=generated["raw_generated_text"])
        del original, skipped, decoded
    return results


def profile_stage(model, processor, samples, protocol, config, output):
    from torch.profiler import profile, ProfilerActivity, record_function
    key, sample = next(iter(samples.items()))
    inputs = inputs_for(processor, key, sample, protocol)
    results = {}
    # One cached decode per path. Prefill is deliberately outside the short trace.
    for label, selected in [("original", []), ("skip", config["layers_1based"])]:
        with SkipPatch(model, selected, config["llama_source_sha256"]):
            pre = prefill(model, inputs)
            token = pre.logits[:, -1:].argmax(-1)
            handles, contexts = [], {}
            def mark(module, name):
                def before(mod, args):
                    context = record_function(name)
                    context.__enter__()
                    contexts[name] = context
                def after(mod, args, out):
                    contexts.pop(name).__exit__(None, None, None)
                handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
            for i, layer in enumerate(model.model.text_model.layers, 1):
                mark(layer, f"L{i:02d}.layer")
                mark(layer.mlp, f"L{i:02d}.mlp")
                for attr in ["gate_proj", "up_proj", "down_proj"]:
                    mark(getattr(layer.mlp, attr), f"L{i:02d}.mlp.{attr}")
            try:
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function(label+"_cached_decode"):
                        decoded = decode(model, pre.past_key_values, inputs, token)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(output/f"{label}_decode_trace.json"))
                events = prof.events()
                names = Counter(e.name for e in events if str(e.device_type).endswith("CPU"))
                mm = Counter()
                for event in events:
                    if event.name not in ["aten::mm", "aten::addmm"]:
                        continue
                    parent = event.cpu_parent
                    while parent:
                        if parent.name.endswith(".mlp") and parent.name.startswith("L"):
                            mm[parent.name] += 1
                            break
                        parent = parent.cpu_parent
                for i in range(1,31):
                    count = 0 if i in selected else 1
                    assert names[f"L{i:02d}.layer"] == 1
                    assert names[f"L{i:02d}.mlp"] == count
                    assert mm[f"L{i:02d}.mlp"] == 3*count, (label,i,mm)
                    for attr in ["gate_proj", "up_proj", "down_proj"]:
                        assert names[f"L{i:02d}.mlp.{attr}"] == count
                results[label] = dict(stable_id=sample["stable_id"], trace_scope="one cached decode, input length 1; prefill outside profiler", markers={k:v for k,v in names.items() if k.startswith("L")},
                                      mlp_matrix_multiply_counts=dict(mm), gpu_events=sum(str(e.device_type).endswith("CUDA") for e in events),
                                      trace_file=str(output/f"{label}_decode_trace.json"))
            finally:
                for h in handles:
                    h.remove()
            del pre, decoded
    return results


def parameter_stage(model, processor, samples, protocol, config, output):
    before = {n:t.detach().clone() for n,t in model.state_dict().items()}
    count_before = sum(p.numel() for p in model.parameters())
    # Parameter identity/data-pointer/version equality proves this patch did not write weights.
    signature = {n:(id(p), p.data_ptr(), p._version, tuple(p.shape)) for n,p in model.named_parameters()}
    selected = config["layers_1based"]
    mlp = {str(i):sum(p.numel() for p in model.model.text_model.layers[i-1].mlp.parameters()) for i in selected}
    norms = {str(i):sum(p.numel() for p in model.model.text_model.layers[i-1].post_attention_layernorm.parameters()) for i in selected}
    with SkipPatch(model, selected, config["llama_source_sha256"]):
        after = model.state_dict()
        assert list(before)==list(after)
        assert sum(p.numel() for p in model.parameters())==count_before
        assert {n:(id(p),p.data_ptr(),p._version,tuple(p.shape)) for n,p in model.named_parameters()}==signature
        assert all(torch.equal(before[n],after[n]) for n in before)
    d, m = model.config.text_config.hidden_size, model.config.text_config.intermediate_size
    assert all(v==3*d*m for v in mlp.values()) and sum(mlp.values())==7962624
    enabled = json.loads((output/"enabled.json").read_text())
    costs={}
    for key,record in enabled["results"].items():
        # Actual lengths observed at an unskipped MLP during real greedy generation.
        shapes=record["generation_input_shapes"]["generate:layer_01.mlp"]
        lengths=[s[1] for s in shapes]
        assert len(lengths)==record["generated_length"] and lengths[0]==record["input_length"] and all(n==1 for n in lengths[1:])
        costs[key]=dict(observed_mlp_input_lengths=lengths, prefill_tokens=lengths[0], decode_calls=len(lengths)-1,
                        decode_tokens=sum(lengths[1:]), projected_flops_omitted_prefill=18*d*m*lengths[0],
                        projected_flops_omitted_decode=18*d*m*sum(lengths[1:]))
    return dict(total_parameters_before=count_before,total_parameters_after=count_before,mlp_parameters=mlp,
                total_skipped_mlp_parameters=sum(mlp.values()),skipped_norm_parameters=norms,
                state_dict_keys_equal=True,parameter_identity_pointer_version_equal=True,state_dict_tensor_values_exact_equal=True,
                hidden_size=d,intermediate_size=m,mlp_projection_flops_per_processed_token=18*d*m,
                costs=costs,scope="theoretical omitted gate/up/down projection FLOPs only, multiply-add=2; excludes norms/activations/attention/vision/head; no speed claim",
                original_weight_bytes=(ROOT/"data/hf_cache/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots"/protocol["model"]["revision"]/"model.safetensors").stat().st_size)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["empty","enabled","profiler","parameters"],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    assert ROOT/"evidence/day1_evening" in output.parents
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/"configs/eval_protocol.json").read_text())
    config=json.loads((ROOT/"configs/llm_ffn_skip.json").read_text())
    report=output/f"{args.stage}.json"
    if report.exists():
        raise ValueError("Existing verification report; choose a new directory instead of overwriting history")
    binding=dict(protocol_sha256=sha(ROOT/"configs/eval_protocol.json"),skip_config_sha256=sha(ROOT/"configs/llm_ffn_skip.json"),
                 code_sha256={f:sha(ROOT/f) for f in ["scripts/ffn_skip.py","scripts/verify_skip.py","scripts/evaluate_skip.py","scripts/run_original.py","scripts/check_scores.py"]})
    prereq={"enabled":"empty","profiler":"enabled","parameters":"enabled"}.get(args.stage)
    if prereq:
        prior=json.loads((output/f"{prereq}.json").read_text())
        assert prior["passed"] and prior["binding"]==binding
    model,processor=load_original_model(output/("runtime_"+args.stage),protocol)
    samples={}
    for key,stem in [("textvqa","textvqa"),("scienceqa","scienceqa_img")]:
        samples[key]=json.loads((ROOT/f"data/manifests/{stem}_validation_256.jsonl").read_text().splitlines()[0])
    stage=dict(empty=empty_stage,enabled=enabled_stage,profiler=profile_stage,parameters=parameter_stage)[args.stage]
    with torch.inference_mode():
        result=stage(model,processor,samples,protocol,config,output)
    atomic(report,dict(stage=args.stage,passed=True,binding=binding,recorded_at=now(),command=sys.argv,results=result))
    print(json.dumps(dict(stage=args.stage,passed=True,report=str(report),results=result),ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
