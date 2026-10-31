"""B2: new-input proofs using unchanged production patches and independent checks."""
import copy
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch as temporary_patch

from evaluate_baseline import atomic, now, sha
from experiment_runtime import (ROOT, BASE, PROTOCOL, NAMES, MANIFESTS, configs, scientific_binding,
                                patch_for, gpu_lock, clean_model, method_signature)
from context_runtime import load_context_model, build_question
from run_original import run_sample
from verify_encoder_topk import Observe, equal, verify_path, signature, toy_checks
import encoder_topk
from encoder_topk import EncoderTopKPatch
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


def sample_run(model, processor, key, row, protocol, output):
    observer = Observe(model)
    try:
        result = run_sample(model, processor, key, ROOT / row["image_file"], build_question(key, row, protocol),
                            sample=row, output_dir=output, protocol=protocol, trace_enabled=False, allow_empty=True)
    finally:
        observer.close()
    return result, observer


class ScoreCapture:
    """Diagnostic-only interception of returned 1D scores, never changes values."""
    def __init__(self, patch):
        self.patch, self.records = patch, []
        self.original = encoder_topk.attention_scores

    def __call__(self, weights, valid):
        scores = self.original(weights, valid)
        positions = self.patch.state["positions"].tolist()
        values = scores.detach().cpu().tolist()
        mask = valid.tolist()
        self.records.append(dict(scores=[[s if v else None for s, v in zip(a, b)] for a, b in zip(values, mask)],
                                 valid=mask, original_positions=positions))
        return scores

    def verify(self):
        stages = self.patch.calls[0]["stages"]
        assert len(self.records) == len(stages)
        for r, stage, rate in zip(self.records, stages, self.patch.rates.values()):
            rebuilt = []
            for scores, valid, positions in zip(r["scores"], r["valid"], r["original_positions"]):
                candidates = [i for i, v in enumerate(valid) if v]
                k = max(1, math.ceil(rate * len(candidates)))
                ranked = sorted(candidates, key=lambda i: (-scores[i], positions[i]))[:k]
                rebuilt.append(sorted(ranked, key=lambda i: positions[i]))
            assert rebuilt == stage["selected_current_indices"]
            r.update(layer=stage["layer"], rate=rate, selected_current_indices=stage["selected_current_indices"],
                     independently_rebuilt=rebuilt, exact_match=True)
        return self.records


def controlled(model, cfg, contract, output):
    hidden = torch.arange(2*16*768, device="cuda", dtype=torch.float32).sin().reshape(2,16,768).to(torch.bfloat16)
    valid = torch.zeros((2,16), device="cuda", dtype=torch.bool)
    valid[0] = True; valid[1,[0,2,4,6,8]] = True
    changed = hidden.clone()
    changed[~valid] += 17 * torch.arange(768, device="cuda").cos().to(torch.bfloat16)
    mask = _prepare_4d_attention_mask(valid, hidden.dtype)
    outputs, reports, tensors = [], [], {}
    for attempt, x in enumerate((hidden, changed)):
        captured, mask_checks, handles = {}, [], []
        # Capture native module outputs, independent of production residual expression.
        for number, layer in enumerate(model.model.vision_model.encoder.layers, 1):
            def pre_attn(mod, args, kwargs, number=number):
                m = kwargs.get("attention_mask")
                captured.setdefault(number, {})["key_valid"] = m[:,0,0,:] == 0
            def post_attn(mod, args, out, number=number):
                kv = captured[number]["key_valid"]
                w = out[1]
                # All query/head rows, invalid key columns only; finite artificial inputs.
                disabled = (~kv)[:,None,None,:].expand_as(w)
                vals = w[disabled]
                assert not vals.numel() or torch.count_nonzero(vals).item() == 0
                mask_checks.append(dict(layer=number, invalid_key_probability_max=float(vals.abs().max()) if vals.numel() else 0., checked_elements=vals.numel()))
                if number in cfg["layers_1based"]:
                    captured[number]["attn"] = out[0].clone()
            handles.append(layer.self_attn.register_forward_pre_hook(pre_attn, with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(post_attn))
            if number in cfg["layers_1based"]:
                def before_layer(mod, args, number=number):
                    captured.setdefault(number, {})["input"] = args[0].clone()
                def after_layer(mod, args, out, number=number):
                    captured[number]["output"] = out.clone()
                def ffn_in(mod, args, number=number):
                    captured[number]["norm2_input"] = args[0].clone()
                def ffn_out(mod, args, out, number=number):
                    captured[number]["mlp_output"] = out.clone()
                handles += [layer.register_forward_pre_hook(before_layer), layer.register_forward_hook(after_layer),
                            layer.layer_norm2.register_forward_pre_hook(ffn_in), layer.mlp.register_forward_hook(ffn_out)]
        try:
            with EncoderTopKPatch(model, cfg, contract) as patch:
                scores = ScoreCapture(patch)
                with temporary_patch.object(encoder_topk, "attention_scores", scores):
                    sparse = model.model.vision_model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state
                    post = model.model.vision_model.post_layernorm(sparse)
                    result = model.model.connector(post)
                score_records = scores.verify()
                final_valid = patch.state["valid"].clone()
                positions = patch.state["positions"].clone()
                trace = copy.deepcopy(patch.calls)
        finally:
            for handle in handles: handle.remove()
        equations = []
        for stage in trace[0]["stages"]:
            number = stage["layer"]; c = captured[number]
            a = c["input"] + c["attn"]
            gathered = a.new_zeros(c["output"].shape)
            for b, indices in enumerate(stage["selected_current_indices"]):
                gathered[b,:len(indices)] = a[b, torch.tensor(indices, device="cuda")]
            layer = model.model.vision_model.encoder.layers[number-1]
            # Independent call to the unchanged native norm2 + MLP after hooks removed.
            branch = layer.mlp(layer.layer_norm2(gathered))
            expected = gathered + branch
            checks = dict(gathered_residual=equal(gathered, c["norm2_input"]),
                          native_branch=equal(branch, c["mlp_output"]), second_residual=equal(expected, c["output"]))
            equations.append(dict(layer=number, **checks))
            if attempt == 0:
                tensors[str(number)] = {"attention_residual":a.cpu(), "gathered":gathered.cpu(), "native_mlp":branch.cpu(),
                                        "actual_layer_output":c["output"].cpu(), "independent_expected":expected.cpu(),
                                        "selected":stage["selected_current_indices"], "valid":torch.tensor(stage["valid_after"])}
        outputs.append((sparse.clone(), post.clone(), result.clone(), final_valid, positions))
        reports.append(dict(controlled_embedded_not_real_image=True, original_valid=[16,5], mask_checks=mask_checks,
                            patch_trace=trace, scores=score_records, second_residual=equations))
    a,b = outputs
    assert torch.equal(a[3],b[3]) and torch.equal(a[4],b[4])
    invariant = dict(encoder_valid=equal(a[0][a[3]],b[0][b[3]]),postnorm_valid=equal(a[1][a[3]],b[1][b[3]]),
                     connector=equal(a[2],b[2]),positions_unchanged=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output.with_suffix(".pt"))
    return dict(attempts=reports,padding_perturbation_invariance=invariant,tensor_file=str(output.with_suffix(".pt")),tensor_sha256=sha(output.with_suffix(".pt")))


def main():
    output = BASE / "b2"; output.mkdir(parents=True, exist_ok=True)
    if (output / "proof.json").exists(): raise ValueError("Completed B2 proof exists")
    protocol = json.loads(PROTOCOL.read_text()); cfgs, contract = configs()
    binding = scientific_binding()
    proof_code = ["scripts/verify_context.py", "scripts/verify_encoder_topk.py", "scripts/verify_skip.py"]
    binding["verification_code_sha256"] = {f:sha(ROOT/f) for f in proof_code}
    atomic(output / "binding.json",binding)
    samples = {json.loads(p.read_text().splitlines()[0])["dataset_key"]:[json.loads(l) for l in p.read_text().splitlines()[:2]] for p in MANIFESTS}
    result = dict(toys=toy_checks(),native={},equivalence={},pruned={},controlled={},skip={})
    with gpu_lock():
        model, processor = load_context_model(output / "runtime",protocol)
        with torch.inference_mode():
            before = signature(model); methods = method_signature(model)
            for key, rows in samples.items():
                for row in rows:
                    rec,obs = sample_run(model,processor,key,row,protocol,output/"native"/key/str(row["row_index"]))
                    result["native"][row["stable_id"]] = dict(input_ids=rec["input_ids"],raw=rec["raw_generated_text"],visual=rec["image_placeholder_token_count"])
                    if row is not rows[0]: continue
                    for name in NAMES[2:]:
                        cfg = copy.deepcopy(cfgs[name]);cfg["keep_rates"] = [1.] * len(cfg["keep_rates"])
                        with EncoderTopKPatch(model,cfg,contract) as patch:
                            got,seen = sample_run(model,processor,key,row,protocol,output/(name+"_100")/key)
                            comparisons = {k:equal(obs.values[k],seen.values[k]) for k in ["encoder","postnorm","connector","prefill_logits"]}
                            assert got["input_ids"]==rec["input_ids"] and got["new_token_ids"]==rec["new_token_ids"]
                            assert patch.calls[0]["bypass"] and patch.calls[0]["score_calls"]==patch.calls[0]["scatter_calls"]==0
                            result["equivalence"][f"{name}/{key}"] = dict(comparisons=comparisons,generated_ids_equal=True,trace=patch.calls)
                        clean_model(model);assert method_signature(model)==methods
            for name in NAMES[2:]:
                for key,rows in samples.items():
                    for row in rows:
                        with patch_for(model,name,cfgs,contract,trace=True) as patch:
                            scores = ScoreCapture(patch)
                            with temporary_patch.object(encoder_topk,"attention_scores",scores):
                                rec,obs = sample_run(model,processor,key,row,protocol,output/name/key/str(row["row_index"]))
                            verify_path(patch.calls,obs)
                            original = result["native"][row["stable_id"]]
                            assert rec["input_ids"]==original["input_ids"] and rec["image_placeholder_token_count"]==original["visual"]
                            assert obs.values["connector"].shape[0]*obs.values["connector"].shape[1]==original["visual"]
                            result["pruned"][f"{name}/{key}/{row['row_index']}"] = dict(trace=copy.deepcopy(patch.calls), observations=obs.events,
                                score_vectors=scores.verify(), input_length=rec["input_length"],visual=original["visual"],score=rec["score"])
                        clean_model(model);assert method_signature(model)==methods
                result["controlled"][name] = controlled(model,cfgs[name],contract,output/f"controlled_{name}")
                atomic(output/f"{name}.json",result["controlled"][name])
            # Reuse the existing Skip checks by replacing only their input builder in this diagnostic context.
            import verify_skip
            first = {k:v[0] for k,v in samples.items()}
            with temporary_patch.object(verify_skip,"build_question",build_question):
                result["skip"]["empty"] = verify_skip.empty_stage(model,processor,first,protocol,cfgs["skip"],output)
                result["skip"]["enabled"] = verify_skip.enabled_stage(model,processor,first,protocol,cfgs["skip"],output)
            clean_model(model);assert method_signature(model)==methods and signature(model)==before
            # Exercise actual exceptional exits, then prove no hooks or method/instance state leaked.
            for name in NAMES[1:]:
                try:
                    with patch_for(model,name,cfgs,contract):
                        raise RuntimeError("controlled cleanup probe")
                except RuntimeError as exc:
                    assert str(exc)=="controlled cleanup probe"
                clean_model(model);assert method_signature(model)==methods and signature(model)==before
            row = first["scienceqa"]
            restored,_ = sample_run(model,processor,"scienceqa",row,protocol,output/"restored_native")
            assert restored["raw_generated_text"]==result["native"][row["stable_id"]]["raw"]
            result["restoration"] = dict(normal_and_exception_exits=True,method_hook_signature_equal=True,weights_identity_pointer_version_equal=True,
                                         native_generation_after_all_switches_equal=True,parameter_count=sum(p.numel() for p in model.parameters()))
    assert scientific_binding()=={k:v for k,v in binding.items() if k!="verification_code_sha256"}
    atomic(output/"proof.json",dict(passed=True,binding=binding,recorded_at=now(),command=sys.argv,results=result))
    print("B2 passed:",output/"proof.json",flush=True)


if __name__=="__main__": main()
