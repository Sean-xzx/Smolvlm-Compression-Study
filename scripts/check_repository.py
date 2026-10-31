"""Read-only navigation/configuration and optional file-integrity checks.

No model/GPU/network access, extraction, execution of scientific helpers, or writes.
Passing these checks does not establish scientific correctness or inference readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = 'd95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d'
ARCHIVE_BYTES = 455733348
PAGES = ['README.md', 'NOTICE.md', 'docs/ASSIGNMENT.zh.md',
         'docs/REQUIREMENTS_MAPPING.zh.md', 'docs/EVIDENCE_INDEX.zh.md',
         'docs/REPRODUCTION.zh.md', 'docs/PUBLICATION.zh.md',
         'docs/SOURCES_AND_CHANGES.zh.md', 'talk/README.zh.md']
CONFIGS = ['baseline', 'encoder_early', 'encoder_middle', 'encoder_late', 'encoder_multi', 'skip']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def anchors(path):
    text = path.read_text()
    found = set(re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)', text))
    used = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith(('```', '~~~')):
            fenced = not fenced
        if fenced or not re.match(r'^#{1,6} ', line):
            continue
        heading = re.sub(r'^#{1,6} ', '', line).strip().lower()
        slug = ''.join(c for c in heading if c in '-_ ' or unicodedata.category(c)[0] in 'LN').replace(' ', '-')
        n = used.get(slug, 0)
        used[slug] = n + 1
        found.add(slug + (f'-{n}' if n else ''))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, help='Optional unpacked postday3-v3 root; check the 19 saved scientific bindings only')
    parser.add_argument('--archive', type=Path, help='Optional existing original scientific tar; read its full SHA without extraction/download')
    args = parser.parse_args()
    errors = []
    result = {'scope': 'reading structure and requested file integrity only',
              'scientific_correctness': 'NOT TESTED', 'inference_readiness': 'NOT TESTED',
              'artifact_root': 'not supplied; complete inputs/dependencies NOT CHECKED',
              'archive': 'not supplied; attachment bytes NOT CHECKED'}
    required = PAGES + ['report/REPORT.zh.pdf', 'report/REPORT.zh.md', 'report/REPORT.zh.docx',
                       'scripts/encoder_topk.py', 'scripts/ffn_skip.py', 'scripts/evaluate_release.py',
                       'requirements-morning.lock.txt', 'configs/eval_protocol_context_v2.json',
                       'figures/quality.csv', 'figures/costs.csv', 'figures/performance.csv']
    required += [f'configs/context_v2/{name}.json' for name in CONFIGS]
    for name in required:
        if not (ROOT/name).is_file():
            errors.append(f'Missing required reading/configuration file: {name}')
    for path in sorted((ROOT/'configs').rglob('*.json')):
        try:
            json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f'Invalid configuration {path.relative_to(ROOT)}: {exc}')
    links = 0
    for name in PAGES:
        page = ROOT/name
        if not page.is_file():
            continue
        for raw in re.findall(r'\]\(([^)]+)\)', page.read_text()):
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue  # Offline check: no claim about remote links.
            target = (page.parent/unquote(url.path)).resolve() if url.path else page
            links += 1
            if not target.exists():
                errors.append(f'{name}: missing link target {raw}')
            elif url.fragment and target.is_file():
                fragment = unquote(url.fragment)
                line = re.fullmatch(r'L(\d+)(?:-L(\d+))?', fragment)
                if line:
                    count = len(target.read_text().splitlines())
                    if not 1 <= int(line[1]) <= int(line[2] or line[1]) <= count:
                        errors.append(f'{name}: out-of-range line link {raw}')
                elif target.suffix == '.md' and fragment not in anchors(target):
                    errors.append(f'{name}: missing heading anchor {raw}')
    result['local_links_checked'] = links
    if args.artifact_root:
        try:
            root = args.artifact_root.resolve(strict=True)
            binding = json.loads((root/'results/context_v2_baseline_20261030/run.json').read_text())['binding']['common']['scientific']
            bound = {}
            for key in ['configurations_sha256', 'manifests_sha256', 'core_code_sha256']:
                bound.update(binding[key])
            if len(bound) != 19:
                raise ValueError('Expected 19 original scientific bindings')
            for rel, expected in bound.items():
                path = (root/rel).resolve()
                if root not in path.parents or sha(path) != expected or sha(ROOT/rel) != expected:
                    raise ValueError(f'Scientific file binding mismatch: {rel}')
            result['artifact_root'] = '19 saved file bindings match; images/model/GPU/scientific behavior NOT CHECKED'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'External scientific root missing/invalid: {exc}')
    if args.archive:
        try:
            if args.archive.stat().st_size != ARCHIVE_BYTES or sha(args.archive) != ARCHIVE_SHA:
                raise ValueError('Original scientific archive size/SHA mismatch')
            result['archive'] = 'original size/SHA match; not extracted or executed'
        except (OSError, ValueError) as exc:
            errors.append(f'External archive missing/invalid: {exc}')
    result.update(requested_checks_passed=not errors, errors=errors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
