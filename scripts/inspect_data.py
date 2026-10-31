"""Inspect pinned HF repositories; retain only one validation diagnostic each."""
import hashlib
import json
import argparse
import random
import io
import sys
import traceback
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder, Image as DatasetImage
from huggingface_hub import snapshot_download
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/diagnostic"
EVIDENCE = ROOT / "evidence/day1_morning"


def main():
    revisions = json.loads((EVIDENCE / "revisions.json").read_text())
    records = {}
    for spec in (s for s in revisions if s["kind"] == "dataset"):
        key, repo, revision = spec["key"], spec["repo_id"], spec["revision"]
        configs = get_dataset_config_names(repo, revision=revision)
        print(key, "configs", configs, flush=True)
        assert configs == ["default"], configs
        builder = load_dataset_builder(repo, name="default", revision=revision)
        info = builder.info
        metadata = {
            "repo_id": repo, "revision": revision, "config_names": configs,
            "selected_config": "default", "features": info.features.to_dict(),
            "splits": {name: {"num_examples": split.num_examples, "num_bytes": split.num_bytes}
                       for name, split in info.splits.items()},
            "access": "HF Datasets streaming; no complete train download or formal sampling",
        }
        (EVIDENCE / f"{key}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        assert "validation" in info.splits
        stream = load_dataset(repo, name="default", split="validation", revision=revision, streaming=True)
        row = next(iter(stream))
        image = row["image"]
        assert image is not None
        image_path = OUT / f"{key}_validation_row0.png"
        image.convert("RGB").save(image_path)
        sample = {k: v for k, v in row.items() if k != "image"}
        sample.update({
            "dataset": repo, "revision": revision, "config": "default", "split": "validation",
            "row_index": 0, "image_file": str(image_path.relative_to(ROOT)),
            "image_size": list(image.size), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "source_decoded_image_mode": image.mode, "saved_image_mode": "RGB",
            "image_save_policy": "Lossless PNG of RGB pixels; no spatial resizing",
            "identity": {"question_id": row["question_id"]} if "question_id" in row else
                        {"revision": revision, "split": "validation", "row_index": 0},
        })
        if key == "textvqa":
            assert len(row["answers"]) == 10
            sample["prompt_policy"] = "image + question; no OCR tokens"
            metadata["ocr_fields_present"] = [k for k in row if "ocr" in k.lower()]
        else:
            assert 0 <= row["answer"] < len(row["choices"])
            sample["prompt_policy"] = "image + question + choices; hint excluded; answer/solution/lecture excluded"
        (OUT / f"{key}_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n")
        (EVIDENCE / f"{key}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print(key, "identity", sample["identity"], "image", image.size,
              "fields", list(row), "question", row["question"], flush=True)
        records[key] = {"metadata_file": f"evidence/day1_morning/{key}_metadata.json",
                        "sample_file": f"data/diagnostic/{key}_sample.json"}
    (EVIDENCE / "dataset_diagnostics.json").write_text(json.dumps(records, indent=2) + "\n")


def prepare_manifests(protocol_path):
    """Download only pinned validation shards, sample indices before reading labels."""
    protocol = json.loads(protocol_path.read_text())
    seed, count = protocol["sampling"]["seed"], protocol["sampling"]["count"]
    target = ROOT / "data/manifests"
    target.mkdir(parents=True, exist_ok=True)
    for key, spec in protocol["datasets"].items():
        stem = "scienceqa_img" if key == "scienceqa" else key
        manifest = target / f"{stem}_validation_{count}.jsonl"
        metadata = manifest.with_suffix(".meta.json")
        indices = sorted(random.Random(seed).sample(range(spec["population_size"]), count))
        binding = dict(dataset=spec, sampling=protocol["sampling"], original_indices=indices)
        cache = ROOT / "data/baseline_inputs" / key
        cache.mkdir(parents=True, exist_ok=True)

        def validate(sample):
            assert sample["revision"] == spec["revision"] and sample["split"] == "validation"
            assert sample["row_index"] in indices and sample["question"].strip()
            path = ROOT / sample["image_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["image_sha256"]
            with PILImage.open(path) as image:
                image.load()
                assert list(image.size) == sample["image_size"] and image.mode == "RGB"
            if key == "textvqa":
                assert len(sample["answers"]) == 10 and all(isinstance(a, str) for a in sample["answers"])
            else:
                assert len(sample["choices"]) >= 2 and 0 <= sample["answer"] < len(sample["choices"])

        if manifest.exists():
            meta = json.loads(metadata.read_text())
            assert meta["binding"] == binding, "Existing manifest uses different sampling/version"
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == meta["manifest_sha256"]
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert len(records) == count and [r["row_index"] for r in records] == indices
            assert len({r["stable_id"] for r in records}) == count
            for sample in records:
                validate(sample)
            print(key, "existing manifest validated", count, flush=True)
            continue
        try:
            snapshot = Path(snapshot_download(spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                                             allow_patterns=["data/validation-*.parquet", "README.md"], max_workers=3))
            shards = sorted(snapshot.glob("data/validation-*.parquet"))
            assert shards, "No validation shards"
            dataset = load_dataset("parquet", data_files={"validation": [str(f) for f in shards]},
                                   split="validation").cast_column("image", DatasetImage(decode=False))
            assert len(dataset) == spec["population_size"]
            diagnostic = json.loads((OUT / f"{key}_sample.json").read_text())
            assert dataset[0]["question"] == diagnostic["question"], "Shard order differs from diagnostic"
            records = []
            for position, index in enumerate(indices):
                record_path = cache / f"row_{index:05d}.json"
                if record_path.exists():
                    sample = json.loads(record_path.read_text())
                    validate(sample)
                else:
                    row = dataset[index]
                    image_data = row["image"]
                    image = PILImage.open(io.BytesIO(image_data["bytes"])) if image_data.get("bytes") is not None else PILImage.open(image_data["path"])
                    source_mode = image.mode
                    image = image.convert("RGB")
                    image_file = cache / f"row_{index:05d}.png"
                    tmp = image_file.with_suffix(".tmp")
                    image.save(tmp, format="PNG")
                    tmp.replace(image_file)
                    sample = {k: v for k, v in row.items() if k != "image"}
                    sample.update(dataset=spec["repo_id"], dataset_key=key, revision=spec["revision"],
                                  config=spec["config"], split=spec["split"], row_index=index,
                                  stable_id=f"{spec['repo_id']}@{spec['revision']}:validation:{index}",
                                  image_file=str(image_file.relative_to(ROOT)), image_size=list(image.size),
                                  image_sha256=hashlib.sha256(image_file.read_bytes()).hexdigest(),
                                  source_decoded_image_mode=source_mode, saved_image_mode="RGB")
                    validate(sample)
                    temp = record_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(sample, ensure_ascii=False) + "\n")
                    temp.replace(record_path)
                records.append(sample)
                if (position + 1) % 32 == 0:
                    print(key, "prepared", position + 1, "/", count, flush=True)
            assert len({r["stable_id"] for r in records}) == count
            if key == "textvqa":
                assert len({r["question_id"] for r in records}) == count
            content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
            temp = manifest.with_suffix(".tmp")
            temp.write_text(content)
            temp.replace(manifest)
            meta = dict(binding=binding, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        count=count, unique_ids=count, images_validated=count, labels_validated=count,
                        preparation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        python=sys.version, shards=[str(f) for f in shards],
                        image_policy="RGB lossless PNG, no spatial resizing", status="complete")
            metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
            print(key, "MANIFEST_READY", manifest, meta["manifest_sha256"], flush=True)
        except Exception:
            (cache / "preparation_error.txt").write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manifests", type=Path, metavar="PROTOCOL")
    args = parser.parse_args()
    if args.prepare_manifests:
        prepare_manifests(args.prepare_manifests)
    else:
        main()
