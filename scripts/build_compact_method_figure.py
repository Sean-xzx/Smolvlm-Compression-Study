"""One compact code/shape schematic for the concise professor report; no inference."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1];F=R/'figures'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(6.7,2.5));ax.axis('off');ax.set_xlim(0,6.7);ax.set_ylim(0,2.5)
def box(x,w,text,color):
 ax.add_patch(FancyBboxPatch((x,1.5),w,.65,boxstyle='round,pad=.03',facecolor=color,edgecolor='#8b99a6',lw=.8));ax.text(x+w/2,1.825,text,ha='center',va='center',fontsize=9.8)
def arrow(x1,x2):ax.annotate('',xy=(x2,1.825),xytext=(x1,1.825),arrowprops={'arrowstyle':'->','color':'#526376','lw':1.2})
box(.05,1.9,'norm1 → Attention\n与输入相加得到a\nAttention长度Na','#edf0f3')
box(2.4,1.8,'head/query 均值打分\nTop-K选集恢复原序\n真实gather(a)','#e5edf7')
box(4.65,2.,'gather(a)进norm2/MLP\n与gather(a)残差相加\n当前Nf，后层变短','#e5edf7');arrow(2.,2.35);arrow(4.25,4.6)
ax.text(.05,2.34,'Top-K：Multi在L3 / L6 / L9依次1024 → 512 → 256 → 128',fontsize=10.5,weight='bold')
ax.text(.05,1.14,'剪枝出口 → 原postnorm → 零回填1024格 → 原pixel shuffle + Linear → LLM',fontsize=9.5)
ax.text(.05,.79,'Text5：13块 → Connector [13,64,576] → 832视觉位置（与基线相同）',fontsize=9.5)
ax.plot([.05,6.65],[.57,.57],color='#d9d9d9',lw=.7)
ax.text(.05,.28,'Skip：L8 / L16 / L24直接返回Attention残差a；省norm2 + MLP，保留cache',fontsize=9.5,color='#17685d')
fig.savefig(F/'08_compact_method.svg',bbox_inches='tight');fig.savefig(F/'08_compact_method.png',bbox_inches='tight',dpi=240);plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=['scripts/encoder_topk.py','scripts/ffn_skip.py','evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4/multi.json']
m=json.loads((F/'figure_manifest.json').read_text());m=[x for x in m if x['id']!='08_compact_method']
m.append(dict(id='08_compact_method',outputs=['08_compact_method.svg','08_compact_method.png'],scientific_version='context-v2',n=1,statistics='code schematic plus TextVQA row5 observed shapes; no attribution or latency estimate',units='tensor shapes / tokens',inputs={p:sha(R/p) for p in inputs},script='scripts/build_compact_method_figure.py',script_sha256=sha(Path(__file__)),output_sha256={n:sha(F/n) for n in ['08_compact_method.svg','08_compact_method.png']}))
(F/'figure_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('Built one compact schematic; existing six scientific charts and all numeric data unchanged')
