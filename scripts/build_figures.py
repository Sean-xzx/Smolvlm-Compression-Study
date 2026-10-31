"""Read-only extraction and scientific plots. Run with --artifact-root U.

Matplotlib is a document-only dependency. It is not installed in the experiment
environment. All scalar chart data are extracted from frozen records.
"""
import argparse, csv, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

P=argparse.ArgumentParser();P.add_argument('--artifact-root',type=Path,required=True)
A=P.parse_args();U=A.artifact_root.resolve();R=Path(__file__).resolve().parents[1];F=R/'figures';F.mkdir(exist_ok=True)
E=U/'evidence/day2_evening/cost_perf_v1';B=U/'evidence/day2_afternoon/20261030_context_v2'
names=['baseline','early','middle','late','multi','skip'];colors=['#68717d','#2575b8','#d68124','#8056aa','#c74747','#258b7c']
inputs={}
def read(p):
    inputs[str(p.relative_to(U))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def csvread(p):
    inputs[str(p.relative_to(U))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open()))
def savecsv(name,rows):
    with (F/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,obj): (F/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

quality=[];qrecords={}
for c in names:
    for d in ['textvqa','scienceqa']:
        rows=[read(p) for p in sorted((U/f'results/context_v2_{c}_20261030/{d}/samples').glob('*.json'))]
        assert len(rows)==256 and all(x['status']=='success' for x in rows)
        qrecords[c,d]=rows
        value=statistics.mean(r['score'] for r in rows)*100
        base=value if c=='baseline' else next(r['score_percent'] for r in quality if r['configuration']=='baseline' and r['dataset']==d)
        quality.append(dict(configuration=c,dataset=d,n=256,score_percent=value,delta_pp=value-base,
                            parse_failure=sum(r['parse_status']=='parse_failure' for r in rows),execution_errors=0,
                            G_mean=statistics.mean(r['generated_length'] for r in rows),
                            G_min=min(r['generated_length'] for r in rows),G_max=max(r['generated_length'] for r in rows),
                            truncated=sum(r['possibly_truncated'] for r in rows)))
savecsv('quality.csv',quality)

costs=[];ranges=[]
cost_fields=['patch_conv','encoder_matrices','vision_with_patch','connector','llm_prefill','llm_decode','output_head','total_major_flops']
for mode in ['actual_G','G16']:
    raw=csvread(E/f'costs/costs_{mode}.csv');assert len(raw)==3072
    for d in ['textvqa','scienceqa']:
        base=[r for r in raw if r['configuration']=='baseline' and r['dataset']==d]
        for c in names:
            rows=[r for r in raw if r['configuration']==c and r['dataset']==d];assert len(rows)==256
            r=dict(mode=mode,configuration=c,dataset=d,n=256)
            for field in cost_fields:r[field+'_GF']=statistics.mean(int(x[field]) for x in rows)/1e9
            r.update(encoder_ratio=r['encoder_matrices_GF']/(statistics.mean(int(x['encoder_matrices']) for x in base)/1e9),
                     total_ratio=r['total_major_flops_GF']/(statistics.mean(int(x['total_major_flops']) for x in base)/1e9))
            costs.append(r)
            ranges.append(dict(mode=mode,configuration=c,dataset=d,B_min=min(int(x['B']) for x in rows),B_max=max(int(x['B']) for x in rows),
                               T_min=min(int(x['T']) for x in rows),T_max=max(int(x['T']) for x in rows),
                               G_min=min(int(x['G']) for x in rows),G_max=max(int(x['G']) for x in rows),
                               total_GF_min=min(int(x['total_major_flops']) for x in rows)/1e9,
                               total_GF_max=max(int(x['total_major_flops']) for x in rows)/1e9))
savecsv('costs.csv',costs);savecsv('ranges.csv',ranges)

lengthraw=csvread(B/'b6/encoder_layer_lengths.csv');assert len(lengthraw)==24576
lengths=[]
for c in names:
    if c in ['baseline','skip']:
        # Native path has no pruning trace: baseline B4 is the direct observation.
        obs=read(B/'b4/baseline.json')['layer_lengths']
        lengths.extend(dict(configuration=c,layer=r['layer'],Na=r['Nattn'],Nf=r['Nffn']) for r in obs)
    else:
        lengths.extend(dict(configuration=c,layer=int(r['layer']),Na=int(r['Nattn']),Nf=int(r['Nffn'])) for r in lengthraw if r['configuration']==c and r['dataset']=='textvqa' and r['row_index']=='5')
savecsv('layer_tokens.csv',lengths)

# Recompute sample medians/IQR from raw measurements and check the old summary.
perf=[];samples=[];measurement_counts=Counter();old=read(E/'performance_summary.json')
for phase in ['natural','first','fixed16','stages']:
    for c in names:
        p=E/phase/c/'measurements.jsonl';inputs[str(p.relative_to(U))]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        for x in rows:measurement_counts['warmup' if x['warmup'] else 'formal']+=1
        groups=defaultdict(list)
        for x in rows:
            if not x['warmup']:groups[(x['sample']['dataset'],x['sample']['row_index'],x.get('stage',x['phase']))].append(x)
        for (d,idx,measure),rs in groups.items():
            assert len(rs)==5
            vs=[x['wall_ms'] for x in rs];q25,q75=np.percentile(vs,[25,75],method='linear')
            samples.append(dict(phase=phase,configuration=c,dataset=d,row_index=idx,measurement=measure,n=5,
                                median_ms=statistics.median(vs),q25_ms=float(q25),q75_ms=float(q75),iqr_ms=float(q75-q25),
                                min_ms=min(vs),max_ms=max(vs),G=rs[0].get('G'),T=rs[0]['sample']['input_length']))
        if phase=='stages':continue
        for d in ['textvqa','scienceqa']:
            ss=[s for s in samples if s['phase']==phase and s['configuration']==c and s['dataset']==d]
            rr=[x for x in rows if x['sample']['dataset']==d and not x['warmup']]
            wall=statistics.mean(s['median_ms'] for s in ss)
            reference=old['groups'][f'{phase}/{c}/{d}/{phase}']
            assert abs(wall-reference['mean_of_sample_median_wall_ms'])<1e-7
            perf.append(dict(phase=phase,configuration=c,dataset=d,sample_count=len(ss),repeats=5,warmups=2,wall_ms=wall,
                             G_mean=statistics.mean(x['G'] for x in rr),
                             peak_allocated_MiB=max(x['peak_allocated_mib'] for x in rr)))
assert measurement_counts==dict(formal=780,warmup=312),measurement_counts
savecsv('performance.csv',perf);savecsv('performance_samples.csv',samples)

matrices=[]
for c in ['baseline','early','middle','late','multi']:
    b=read(B/f'b4/{c}.json');layers={'baseline':[3,6,9],'early':[3,4],'middle':[6,7],'late':[9,10],'multi':[3,4,6,7,9,10]}[c]
    for l in layers:
        for f in ['fc1','fc2']:
            label=f'Vision.L{l:02d}.mlp.{f}'
            ops=[o for o in b['matrix_operators'] if o['marker']==label and o['operator']=='aten::addmm'];assert len(ops)==1
            o=ops[0]
            matrices.append(dict(configuration=c,layer=l,module=f,prehook=str(b['module_input_shapes'][label]),
                                 matrix_inputs=str(o['input_shapes'][1:3]),cpu_us=o['cpu_us'],cuda_us=o['cuda_us']))
savecsv('matrix_evidence.csv',matrices)
cfgs={c:read(U/'configs/context_v2'/('encoder_'+c+'.json' if c not in ['baseline','skip'] else c+'.json')) for c in names}
protocol=read(U/'configs/eval_protocol_context_v2.json')
for p in ['encoder_topk.py','ffn_skip.py','context_runtime.py','run_original.py']:
    pp=U/'scripts'/p;inputs[str(pp.relative_to(U))]=hashlib.sha256(pp.read_bytes()).hexdigest()
dump('source_inputs.json',inputs)
dump('report_data.json',dict(quality=quality,costs=costs,ranges=ranges,lengths=lengths,performance=perf,performance_samples=samples,matrices=matrices,configs=cfgs,protocol=protocol,old_measurement_counts=dict(measurement_counts)))

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':15,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
manifest=[]
def save(fig,stem,fields,n,stat,unit,version):
    fig.savefig(F/(stem+'.svg'),bbox_inches='tight');fig.savefig(F/(stem+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
    manifest.append(dict(id=stem,outputs=[stem+'.svg',stem+'.png'],fields=fields,n=n,statistics=stat,units=unit,scientific_version=version,
                         input_manifest='source_inputs.json',input_manifest_sha256=hashlib.sha256((F/'source_inputs.json').read_bytes()).hexdigest(),
                         script='scripts/build_figures.py',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

fig,axes=plt.subplots(1,2,figsize=(12.5,4.2),layout='constrained')
for ax,d,title in zip(axes,['textvqa','scienceqa'],['TextVQA 十参考留一软分','ScienceQA-IMG 选项准确率']):
    vals=[next(x['score_percent'] for x in quality if x['configuration']==c and x['dataset']==d) for c in names]
    bars=ax.bar(names,vals,color=colors,edgecolor='white');ax.set_ylim(0,80);ax.set_ylabel('%');ax.set_title(title+'  n=256')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',rotation=25)
save(fig,'03_quality',['score','configuration','dataset'],256,'mean of 256 saved sample scores; parse failures retain denominator','percent','context-v2 3072')

fig,axes=plt.subplots(2,1,figsize=(11.5,6),sharex=True,layout='constrained')
for j,field in enumerate(['Na','Nf']):
    for c,color in zip(names,colors):
        rows=[r for r in lengths if r['configuration']==c]
        axes[j].step([r['layer'] for r in rows],[r[field] for r in rows],where='mid',label=c,color=color,ls='--' if c=='skip' else '-',marker={'baseline':'o','early':'^','middle':'s','late':'D','multi':'v','skip':'x'}[c],markersize=4,alpha=.85)
    axes[j].set_ylim(0,1150);axes[j].set_yticks([128,256,512,1024]);axes[j].set_ylabel(field+' / 每块');axes[j].grid(alpha=.18)
axes[0].set_title('Attention 输入 Na 与当前 FFN 输入 Nf 分开计数');axes[0].legend(ncol=6,fontsize=10,loc='lower left')
axes[1].set_xticks(range(1,13));axes[1].set_xlabel('视觉 Encoder 层号（1 起）')
save(fig,'02_tokens',['layer','Nattn','Nffn'],1,'TextVQA row5 direct observed per-block lengths; baseline=skip','tokens/block','context-v2 b4/b6')

fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),layout='constrained')
for ax,d in zip(axes,['textvqa','scienceqa']):
    rows=[next(x for x in costs if x['mode']=='G16' and x['dataset']==d and x['configuration']==c) for c in names]
    x=np.arange(6);ax.bar(x-.18,[r['encoder_ratio'] for r in rows],width=.36,color=colors,edgecolor='#25303f',label='Encoder 本体')
    ax.bar(x+.18,[r['total_ratio'] for r in rows],width=.36,color=colors,hatch='//',alpha=.65,label='全模型 G*=16')
    ax.set_xticks(x,names,rotation=25);ax.set_ylim(0,1.17);ax.set_ylabel('相对同配置版本 baseline 的成本比');ax.set_title(d+'  n=256 离线估算');ax.legend(fontsize=10);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
save(fig,'04_cost',['encoder_matrices','total_major_flops','G'],256,'ratio of per-dataset arithmetic means; G*=16 offline only','baseline ratio','cost-perf-v1')

fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
for ax,d in zip(axes,['textvqa','scienceqa']):
    x=np.arange(6)
    for c,col,xx in zip(names,colors,x):
        ss=[s for s in samples if s['phase']=='fixed16' and s['configuration']==c and s['dataset']==d];assert len(ss)==1
        s=ss[0];ax.bar(xx,s['median_ms'],color=col,edgecolor='white')
        ax.errorbar(xx,s['median_ms'],yerr=[[s['median_ms']-s['q25_ms']],[s['q75_ms']-s['median_ms']]],color='#202b36',capsize=4)
        ax.text(xx,s['q75_ms']+12,f'{s["median_ms"]:.1f}',ha='center',fontsize=10)
    ax.set_ylim(0,520);ax.set_xticks(x,names,rotation=25);ax.set_ylabel('同步主机 wall / ms');ax.set_title(d+'  每集1题 × 5正式重复');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
save(fig,'05_fixed16',['wall_ms','warmup','sample','G'],1,'median and IQR of 5 repeats; 2 warmups; all G=16','milliseconds','cost-perf-v1')

fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
for ax,d in zip(axes,['textvqa','scienceqa']):
    for xx,c,col in zip(range(6),names,colors):
        r=next(x for x in perf if x['phase']=='natural' and x['configuration']==c and x['dataset']==d)
        ax.bar(xx,r['wall_ms'],color=col);ax.text(xx,r['wall_ms']+8,f'{r["wall_ms"]:.1f}\nG={r["G_mean"]:g}',ha='center',fontsize=10)
    ax.set_xticks(range(6),names,rotation=25);ax.set_ylim(0,390);ax.set_ylabel('逐题 median 的均值 / ms');ax.set_title(d+'  每集4题 × 5正式重复');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
save(fig,'06_natural',['wall_ms','G','warmup','sample'],4,'arithmetic mean of 4 per-sample five-repeat medians; 2 warmups','milliseconds','cost-perf-v1')

# Schematic, explicitly sourced from code and recorded Text5 shapes. No attention
# attribution/heatmap is generated or claimed.
fig,ax=plt.subplots(figsize=(12.5,6));ax.set_xlim(0,12.5);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,text,color='#eaf0f6'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#8b99a6',linewidth=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arr(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#526376',lw=1.5))
ax.text(.05,5.7,'Encoder Top-K 与原 Connector 的真实传播',fontsize=18,weight='bold')
box(.1,4.1,2.4,1,'x + Attention(norm1(x))\n当前层 Attention 用 Na')
box(3,4.1,2.35,1,'head / query 均值\nTop-K 并恢复原序','#e5edf7')
box(5.85,4.1,2.45,1,'gather(a)\n当前 FFN 用 Nf','#e5edf7')
box(8.8,4.1,3.4,1,'gather(a) + MLP(norm2(...))\n后层 Attention 用短序列')
for x in [2.5,5.35,8.3]:arr(x,4.6,x+.5,4.6)
box(.1,2.2,2.4,1,'Text5 独立13图块\n每块 1024 × 768')
box(3,2.2,2.35,1,'Multi L3 / L6 / L9\n512 / 256 / 128','#fae9e9')
box(5.85,2.2,2.45,1,'postnorm 后零回填\n每块恢复1024格')
box(8.8,2.2,3.4,1,'原 pixel shuffle 4×4\n64 × 12288，再 Linear 到576')
for x in [2.5,5.35,8.3]:arr(x,2.7,x+.5,2.7)
box(.1,.3,5.25,1,'LLM 仍接收 13 × 64 = 832 视觉位置\n本样本总输入 T=889，Encoder少算未缩短LLM序列')
box(5.85,.3,6.35,1,'FFN Skip L8 / L16 / L24\n保留 a = x + Attention(norm1(x))，省 norm2 + MLP','#e4f3ef')
save(fig,'01_method',['encoder_topk.py','ffn_skip.py','Text5 input shapes'],1,'code schematic and observed shapes; no causal attribution','shape / tokens','context-v2')
dump('figure_manifest.json',manifest)
print('Extracted 3072 quality, 6144 cost, 24576 lengths, 780 formal + 312 warmup; 6 figures')
