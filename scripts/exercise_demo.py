"""CPU-only 75% retention exercise using the exact frozen selection helper.
This is a configuration/helper demonstration, not new model quality evidence.
"""
import argparse,json,sys,hashlib,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
U=a.artifact_root.resolve();out=a.output_dir.resolve()
if out==U or U in out.parents:p.error('Use a new exercise directory outside scientific root')
if out.exists() and any(out.iterdir()):p.error('New empty output directory required')
out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(U/'scripts'))
import torch
from encoder_topk import select_tokens,restore_grid
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=U/'configs/context_v2/encoder_early.json';before=sha(parent)
cfg=dict(kind='exercise-only-not-frozen-protocol',parent_relative_path=str(parent.relative_to(U)),parent_sha256=before,
         layers_1based=[3],keep_rates=[0.75],note='No production binding/hash is changed; CPU helper fixture only')
(out/'exercise_early75.json').write_text(json.dumps(cfg,indent=2)+'\n')
hidden=torch.arange(1024*768,dtype=torch.float32).reshape(1,1024,768)
positions=torch.arange(1024).reshape(1,1024);valid=torch.ones(1,1024,dtype=torch.bool)
scores=torch.arange(1024,dtype=torch.float32).reshape(1,1024)
result,pos,mask,selected,ks=select_tokens(hidden,positions,valid,scores,cfg['keep_rates'][0])
assert ks==[768] and list(result.shape)==[1,768,768]
assert torch.equal(pos,torch.arange(256,1024).reshape(1,-1))
grid,occupied=restore_grid(result,pos,mask,1024)
assert torch.equal(grid[:,256:],hidden[:,256:]) and torch.count_nonzero(grid[:,:256])==0
assert sha(parent)==before
report=dict(passed=True,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=sys.argv,cwd=str(Path.cwd()),
            configuration_sha256=sha(out/'exercise_early75.json'),helper_sha256=sha(U/'scripts/encoder_topk.py'),
            device='cpu',input_shape=list(hidden.shape),output_shape=list(result.shape),K=ks,restored_shape=list(grid.shape),
            selected_original_positions_range=[int(pos.min()),int(pos.max())],original_config_unchanged=True,
            scope='AI-run CPU deterministic selection/grid helper fixture; no model forward, quality/latency or user independent exercise claim')
(out/'exercise_demo.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
