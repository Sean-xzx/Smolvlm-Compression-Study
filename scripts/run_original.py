"""Minimal HF README adaptation: original weights, eager CUDA, read-only trace.

Source: HuggingFaceTB/SmolVLM-256M-Instruct README at the revision in
evidence/day1_morning/revisions.json. No pruning, skipping, training or quantization.
"""
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import urllib.request

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from check_scores import score_response

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "evidence/day1_morning"
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_URL = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def shapes(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, Mapping):
        return {k: shapes(v) for k, v in value.items() if isinstance(v, (torch.Tensor, dict, tuple, list))}
    if isinstance(value, (tuple, list)):
        return [shapes(v) for v in value]
    return None


def install_trace(model):
    trace, handles = {}, []
    modules = {
        "vision_embeddings": model.model.vision_model.embeddings,
        "encoder": model.model.vision_model.encoder,
        "connector": model.model.connector,
        "projection_linear": model.model.connector.modality_projection.proj,
        "llm": model.model.text_model,
    }
    modules.update({f"encoder_layer_{i+1}": layer for i, layer in enumerate(model.model.vision_model.encoder.layers)})
    for key, module in modules.items():
        trace[key] = {"calls": 0, "class": type(module).__name__,
                      "source_file": inspect.getsourcefile(type(module)),
                      "forward_line": inspect.getsourcelines(type(module).forward)[1]}

        def pre(mod, args, kwargs, key=key):
            entry = trace[key]
            entry["calls"] += 1
            if entry["calls"] == 1:
                entry["first_input_args"] = shapes(args)
                entry["first_input_kwargs"] = shapes(kwargs)
                if key == "vision_embeddings" and kwargs.get("patch_attention_mask") is not None:
                    mask = kwargs["patch_attention_mask"]
                    entry["valid_patch_tokens_per_block"] = mask.flatten(1).sum(1).cpu().tolist()

        def post(mod, args, kwargs, output, key=key):
            if trace[key]["calls"] == 1:
                trace[key]["first_output"] = shapes(output)

        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(post, with_kwargs=True))
    return trace, handles


def run_sample(model, processor, key, image_path, question, sample=None, repeat=False, *,
               output_dir, trace_enabled=True, protocol=None, persist=True, allow_empty=False):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
    input_shapes = shapes(inputs)
    n = inputs["input_ids"].shape[1]
    pixel_values = inputs["pixel_values"]
    flat = pixel_values.flatten(0, 1)
    valid = flat.ne(0).any(dim=(1, 2, 3))
    trace, handles = install_trace(model) if trace_enabled else ({}, [])
    generation = dict(protocol["generation"]) if protocol else dict(max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True)
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    new_ids = output[:, n:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos or []
    ended_on_eos = bool(new_ids.shape[1] and int(new_ids[0, -1].item()) in eos_ids)
    if not allow_empty:
        assert raw.strip(), "Empty model response (diagnostic mode)"
    placeholder_count = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    if trace_enabled:
        projected_shape = trace["connector"]["first_output"]["shape"]
        assert placeholder_count == projected_shape[0] * projected_shape[1]
        assert trace["llm"]["first_input_kwargs"]["inputs_embeds"]["shape"][1] == n
    record = {
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "key": key, "model_id": MODEL_ID, "model_revision": model.config._commit_hash,
        "processor_revision": REVISION, "image_file": str(image_path.relative_to(ROOT)),
        "image_source": IMAGE_URL if key == "official" else sample["dataset"],
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "image_width_height": list(image.size), "question_for_model": question, "messages": messages,
        "prompt": prompt, "input_shapes": input_shapes, "input_ids": inputs["input_ids"].tolist(),
        "input_length": n, "new_token_ids": new_ids.tolist(), "generated_length": new_ids.shape[1],
        "max_new_tokens": generation["max_new_tokens"], "reached_length_limit": new_ids.shape[1] == generation["max_new_tokens"],
        "ended_on_eos": ended_on_eos,
        "possibly_truncated": new_ids.shape[1] == generation["max_new_tokens"] and not ended_on_eos,
        "raw_generated_text": raw, "device": str(model.device), "dtype": str(model.dtype),
        "attention_backend": model.config._attn_implementation,
        "valid_image_blocks": int(valid.sum().item()), "total_image_block_slots": int(valid.numel()),
        "padded_image_blocks": int((~valid).sum().item()),
        "image_placeholder_token_count": placeholder_count,
        "projected_tokens_match_placeholders": True if trace_enabled else None,
        "llm_first_input_matches_full_prompt_length": True if trace_enabled else None,
        "generation_overrides": generation, "trace_enabled": trace_enabled, "trace": trace,
    }
    if repeat:
        with torch.inference_mode():
            again = model.generate(**inputs, **generation)
            torch.cuda.synchronize()
        record["repeat"] = {"token_ids_equal": torch.equal(output, again),
                            "new_token_ids": again[:, n:].tolist(),
                            "raw_generated_text": processor.batch_decode(again[:, n:], skip_special_tokens=True)[0],
                            "hooks_removed": True}
        assert record["repeat"]["token_ids_equal"], "Repeated deterministic generation differs"
    if sample:
        record["sample"] = sample
        record.update(score_response(key, raw, sample))
    if persist:
        save(OUT / f"{key}_result.json", record)
        (OUT / f"{key}_prompt.txt").write_text(prompt)
        (OUT / f"{key}_raw_output.txt").write_text(raw)
    print(key, "input", n, "generated", new_ids.shape[1], "output", repr(raw),
          "score", record.get("score"), flush=True)
    return record


def build_question(key, sample, protocol=None):
    question = sample["question"]
    if key == "scienceqa":
        options = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample["choices"]))
        question += "\n" + options
    instruction = (protocol["prompts"][key]["instruction"] if protocol else
                   "Answer with only the option letter." if key == "scienceqa" else
                   "Answer with a short word or phrase.")
    return question + "\n" + instruction


def load_original_model(output_dir, protocol=None):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    if protocol:
        required = dict(device="cuda:0", dtype="bfloat16", attention_backend="eager",
                        batch_size=1, eval=True, inference_mode=True,
                        deterministic_algorithms=True, allow_tf32=False)
        for key, value in required.items():
            assert protocol[key] == value, (key, "unsupported protocol setting")
        assert protocol["model"]["repo_id"] == MODEL_ID
        assert not protocol["prompts"]["textvqa"]["use_ocr"]
        assert not protocol["prompts"]["scienceqa"]["use_hint"]
        assert protocol["scoring"]["code_sha256"] == hashlib.sha256((ROOT / "scripts/check_scores.py").read_bytes()).hexdigest()
        assert protocol["scoring"]["mmf_sha256"] == hashlib.sha256((ROOT / "scripts/vendor/mmf_m4c_evaluators.py").read_bytes()).hexdigest()
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    assert torch.cuda.is_bf16_supported(), "bfloat16 check required"
    torch.manual_seed(protocol["generation_seed"] if protocol else 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    assert model.config._commit_hash == REVISION
    weight_path = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION, local_files_only=True))
    expected_weight = next(f for f in json.loads((PROVENANCE / "sources/model_files_metadata.json").read_text())["siblings"]
                           if f["rfilename"] == "model.safetensors")
    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    assert weight_sha == expected_weight["lfs"]["sha256"], "Downloaded weight checksum mismatch"
    for cfg in [model.config, model.config.vision_config, model.config.text_config]:
        assert cfg._attn_implementation == "eager", cfg._attn_implementation
    save(OUT / "runtime_model_config.json", model.config.to_dict())
    save(OUT / "runtime_generation_config.json", model.generation_config.to_dict())
    save(OUT / "runtime_processor_config.json", processor.to_dict())
    processor.save_pretrained(OUT / "processor_snapshot")
    modules = [{"name": name, "class": type(module).__name__} for name, module in model.named_modules()]
    save(OUT / "model_modules.json", modules)
    source_file = Path(inspect.getsourcefile(type(model)))
    (OUT / "sources/installed_modeling_idefics3.py").write_bytes(source_file.read_bytes())
    save(OUT / "model_runtime.json", {
        "model_class": type(model).__name__, "source_file": str(source_file),
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vision_layers": len(model.model.vision_model.encoder.layers),
        "llm_layers": len(model.model.text_model.layers), "batch_size": 1,
        "eval_mode": not model.training, "deterministic_algorithms": True,
        "seed": protocol["generation_seed"] if protocol else 0,
        "weight_file": str(weight_path), "weight_file_bytes": weight_path.stat().st_size,
        "weight_sha256": weight_sha, "weight_matches_upstream_lfs_sha256": True,
    })
    if protocol:
        assert protocol["model"]["revision"] == REVISION
        actual = processor.image_processor.to_dict()
        for key, value in protocol["preprocessing"].items():
            if key == "processor_class":
                assert type(processor).__name__ == value
                continue
            assert actual.get(key) == value, (key, actual.get(key), value)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", choices=["official", "datasets", "all"], default="official")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    model, processor = load_original_model(args.output_dir, protocol)
    if args.samples in ("official", "all"):
        path = ROOT / "data/diagnostic/official_statue.jpg"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(IMAGE_URL, timeout=60).read())
        run_sample(model, processor, "official", path, "Can you describe this image?", repeat=True, output_dir=args.output_dir, trace_enabled=not args.no_trace, protocol=protocol)
    if args.samples in ("datasets", "all"):
        for key in ["textvqa", "scienceqa"]:
            sample = json.loads((ROOT / f"data/diagnostic/{key}_sample.json").read_text())
            question = build_question(key, sample, protocol)
            run_sample(model, processor, key, ROOT / sample["image_file"], question, sample=sample,
                       output_dir=args.output_dir, trace_enabled=not args.no_trace, protocol=protocol)



REVISION = next(s["revision"] for s in json.loads((PROVENANCE / "revisions.json").read_text()) if s["key"] == "model")

if __name__ == "__main__":
    main()
