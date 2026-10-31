"""Read frozen quality artifacts, write independent major-matrix/conv cost tables.

No inference or old report-writing helpers are invoked. Non-matrix arithmetic,
transcendentals, comparisons and movement are reported separately from totals.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from evaluate_baseline import atomic,sha,now

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'evidence/day2_evening/cost_perf_v1'
NAMES=['baseline','skip','early','middle','late','multi']


def write_csv(path, rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def components(record,name,g,config):
    v,t=config['vision_config'],config['text_config'];dv,mv=v['hidden_size'],v['intermediate_size'];d,m=t['hidden_size'],t['intermediate_size']
    h=v['num_attention_heads'];dkv=t['num_key_value_heads']*t['head_dim'];V=t['vocab_size'];layers=t['num_hidden_layers'];sf=config['scale_factor']
    B=record['valid_image_blocks'];T=record['input_length'];N0=(v['image_size']//v['patch_size'])**2;n=g-1
    patch=2*B*N0*(v['num_channels']*v['patch_size']**2)*dv
    connector=2*B*(N0//sf**2)*(dv*sf**2)*d
    if name in ['baseline','skip']:lengths=[dict(layer=i,attention_physical_length=N0,ffn_physical_length=N0) for i in range(1,v['num_hidden_layers']+1)];stages=[]
    else:lengths=record['encoder_computation']['layer_lengths'];stages=record['encoder_computation']['stages']
    encoder=0;score=0;cast=0;sorts=0;gelu_elements=0;bias=B*N0*dv;vision_softmax_logits=0;vision_norm_rows=B*N0 # postnorm differs for pruning below
    per_layer=[]
    for l in lengths:
        na,nf=l['attention_physical_length'],l['ffn_physical_length']
        value=8*B*na*dv**2+4*B*na**2*dv+4*B*nf*dv*mv
        encoder+=value;gelu_elements+=B*nf*mv;bias+=4*B*na*dv+B*nf*(mv+dv)
        vision_softmax_logits+=B*h*na**2
        per_layer.append(dict(layer=l['layer'],B=B,Na=na,Nf=nf,flops=value))
    for stage in stages:
        na=stage['attention_shape'][-1];score+=B*na*na*(h+2);cast+=B*h*na*na;sorts+=3*B
    active=layers-3 if name=='skip' else layers
    projection=4*d*(d+dkv);ffn=6*d*m;keys=n*T+n*(n+1)//2
    pf_proj=layers*T*projection;pf_attention=layers*4*T*T*d;pf_ffn=active*T*ffn
    dec_proj=layers*n*projection;dec_attention=layers*4*keys*d;dec_ffn=active*n*ffn
    head_prefill=2*d*V;head_decode=n*2*d*V
    result=dict(patch_conv=patch,encoder_matrices=encoder,vision_with_patch=patch+encoder,connector=connector,
                prefill_projections=pf_proj,prefill_attention=pf_attention,prefill_ffn=pf_ffn,
                llm_prefill=pf_proj+pf_attention+pf_ffn,decode_projections=dec_proj,decode_attention=dec_attention,decode_ffn=dec_ffn,
                llm_decode=dec_proj+dec_attention+dec_ffn,head_prefill=head_prefill,head_decode=head_decode,output_head=head_prefill+head_decode)
    result['total_major_flops']=patch+encoder+connector+result['llm_prefill']+result['llm_decode']+result['output_head']
    # Arithmetic conventions are explicit, algorithmic and not hardware instruction counts.
    result.update(score_fp32_arithmetic=score,score_cast_elements=cast,score_cast_read_write_bytes=cast*6,
                  stable_argsort_calls=sorts,vision_gelu_evaluations=gelu_elements,vision_bias_additions=bias,
                  llm_silu_evaluations=active*(T+n)*m,llm_gated_multiplications=active*(T+n)*m,
                  llm_rmsnorm_rows=(layers+active+1)*(T+n),
                  llm_rmsnorm_basic_arithmetic=(layers+active+1)*(T+n)*(4*d+1),
                  llm_rmsnorm_rsqrt_evaluations=(layers+active+1)*(T+n),
                  softmax_exp_evaluations=vision_softmax_logits+layers*t['num_attention_heads']*(T*T+keys))
    return result,per_layer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--probe-root',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output_dir.resolve()
    if ROOT/'evidence/day2_afternoon' in out.parents:raise ValueError('Old evidence is read-only')
    if out.exists() and any(out.iterdir()):raise ValueError('New analysis output directory required')
    out.mkdir(parents=True,exist_ok=True)
    config_path=ROOT/'results/context_v2_baseline_20261030/runtime/runtime_model_config.json';config=json.loads(config_path.read_text())
    v,t=config['vision_config'],config['text_config'];assert (v['hidden_size'],v['intermediate_size'],v['patch_size'],t['hidden_size'],t['intermediate_size'],t['num_key_value_heads'],t['head_dim'])==(768,3072,16,576,1536,3,64)
    probes={}
    for name in NAMES:
        assert json.loads((args.probe_root/f'probe/{name}/complete.json').read_text())['passed']
        for dataset in ['textvqa','scienceqa']:
            p=args.probe_root/f'probe/{name}/probe_{dataset}.json';proof=json.loads(p.read_text());assert proof['passed'] and proof['all_hooks_removed']
            assert proof['counts']['vision']==proof['counts']['connector']==1
            assert all(s==[1,1,t['hidden_size']] for s in proof['lm_head_inputs'])
            assert proof['attention_q_s'][0]['q']==proof['sample']['input_length']
            shapes=proof['operator_shapes'];B=proof['sample']['valid_image_blocks'];T=proof['sample']['input_length']
            assert shapes['model.vision_model.embeddings.patch_embedding']['input']==[B,3,512,512]
            assert shapes['model.vision_model.embeddings.patch_embedding']['weight']==[768,3,16,16]
            assert shapes['model.connector.modality_projection.proj']['input']==[B,64,12288]
            assert shapes['model.connector.modality_projection.proj']['weight']==[576,12288]
            assert shapes['model.text_model.layers.0.self_attn.q_proj']['weight']==[576,576]
            assert shapes['model.text_model.layers.0.self_attn.k_proj']['weight']==[192,576]
            assert shapes['model.text_model.layers.0.self_attn.v_proj']['weight']==[192,576]
            assert shapes['model.text_model.layers.0.mlp.up_proj']['weight']==[1536,576]
            assert shapes['model.text_model.layers.0.mlp.down_proj']['weight']==[576,1536]
            assert shapes['lm_head']['weight']==[49280,576]
            # Independent sum of four measured Linear input/weight matrix sizes.
            measured_projection=sum(2*T*shapes[f'model.text_model.layers.0.self_attn.{attr}']['weight'][0]*shapes[f'model.text_model.layers.0.self_attn.{attr}']['weight'][1] for attr in ['q_proj','k_proj','v_proj','o_proj'])
            assert measured_projection==4*T*576*(576+192)
            probes[str(p.relative_to(ROOT))]=sha(p)
    old_csv=ROOT/'evidence/day2_afternoon/20261030_context_v2/b6/encoder_layer_lengths.csv'
    with old_csv.open() as stream:old={(r['configuration'],r['dataset'],int(r['row_index']),int(r['layer'])):int(r['matmul_flops_estimate']) for r in csv.DictReader(stream)}
    actual=[];normalized=[];quality_hashes={};crosschecked=0
    for name in NAMES:
        for dataset in ['textvqa','scienceqa']:
            paths=sorted((ROOT/f'results/context_v2_{name}_20261030/{dataset}/samples').glob('row_*.json'));assert len(paths)==256
            for p in paths:
                r=json.loads(p.read_text());assert r['status']=='success';quality_hashes[str(p.relative_to(ROOT))]=sha(p)
                identity=dict(configuration=name,dataset=dataset,row_index=r['row_index'],stable_id=r['stable_id'],B=r['valid_image_blocks'],T=r['input_length'],actual_G=r['generated_length'])
                for g,target in [(r['generated_length'],actual),(16,normalized)]:
                    costs,lengths=components(r,name,g,config);target.append(dict(**identity,G=g,**costs))
                    if name not in ['baseline','skip'] and target is actual:
                        for l in lengths:
                            assert l['flops']==old[(name,dataset,r['row_index'],l['layer'])];crosschecked+=1
    assert len(actual)==len(normalized)==3072 and crosschecked==24576
    by={(r['configuration'],r['dataset'],r['row_index']):r for r in normalized};skip_checks=0
    for r in normalized:
        baseline=by[('baseline',r['dataset'],r['row_index'])]
        if r['configuration']=='skip':
            assert r['vision_with_patch']==baseline['vision_with_patch'] and r['connector']==baseline['connector']
            assert baseline['total_major_flops']-r['total_major_flops']==18*t['hidden_size']*t['intermediate_size']*(r['T']+15);skip_checks+=1
        if r['configuration'] in ['early','middle','late','multi']:
            assert r['llm_prefill']==baseline['llm_prefill'] and r['llm_decode']==baseline['llm_decode'] and r['output_head']==baseline['output_head'] and r['connector']==baseline['connector']
    summaries={}
    for mode,rows in [('actual_G',actual),('G16',normalized)]:
        groups={}
        for name in NAMES:
            for ds in ['textvqa','scienceqa']:
                subset=[r for r in rows if r['configuration']==name and r['dataset']==ds]
                keys=['encoder_matrices','vision_with_patch','patch_conv','connector','llm_prefill','llm_decode','output_head','total_major_flops','score_fp32_arithmetic','score_cast_read_write_bytes','stable_argsort_calls']
                groups[name+'/'+ds]=dict(count=256,components={k:dict(mean=sum(r[k] for r in subset)/256,min=min(r[k] for r in subset),max=max(r[k] for r in subset),sum=sum(r[k] for r in subset)) for k in keys})
        for name in NAMES:
            for ds in ['textvqa','scienceqa']:
                g=groups[name+'/'+ds];base=groups['baseline/'+ds]
                g['total_ratio_to_baseline']=g['components']['total_major_flops']['mean']/base['components']['total_major_flops']['mean']
                g['encoder_ratio_to_baseline']=g['components']['encoder_matrices']['mean']/base['components']['encoder_matrices']['mean']
        summaries[mode]=groups
    write_csv(out/'costs_actual_G.csv',actual);write_csv(out/'costs_G16.csv',normalized)
    atomic(out/'source_quality_sha256.json',quality_hashes)
    atomic(out/'summary.json',dict(passed=True,at=now(),name='full-model major matrix/conv FLOPs estimate; multiply-add=2',groups=summaries,
       checks=dict(encoder_rows_exact_to_old_csv=crosschecked,skip_projection_difference_checks=skip_checks,fixed_G_llm_connector_equal=True,head_and_vision_call_scope_empirically_confirmed=True),
       binding=dict(script_sha256=sha(Path(__file__)),model_config_sha256=sha(config_path),old_encoder_csv_sha256=sha(old_csv),probes=probes,performance_config_sha256=sha(ROOT/'configs/cost_perf_v1.json')),
       scope=dict(main='patch convolution + all vision QKV/O,QK/AV,fc1/fc2 + original Connector Linear + LLM GQA projections/dense attention/gated FFN prefill and G-1 cached decodes + one-position head each call',
           auxiliary='FP32 score arithmetic B*N^2*(H+2), division counted1; casts6 bytes/element (BF16 read + FP32 write), 3 stable sorts/block/stage, bias adds, activation evaluation counts, LLM gating multiply and RMS basic arithmetic/rsqrt, dense softmax exp count',
           excluded_from_main='bias/norm/gelu/silu/softmax/rope/mask/position/score/guards/comparisons/sorting/gather/scatter/CPU loops and synchronization/KV movement/token lookup/sampling/control/preprocess/H2D/I/O; aux convention counts are not added to matrix total or claimed hardware instructions',
           standardized_G='G*=16 offline standardized workload, not 512 additional fixed-length inference runs')))
    print('Cost tables passed: 3072 actual + 3072 standardized rows; 24576 encoder checks; 512 Skip difference checks',flush=True)


if __name__=='__main__':main()
