"""Final local acceptance. No model load or mutation of scientific assets."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,re,subprocess,sys,zipfile
from pypdf import PdfReader
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
started=datetime.datetime.now(datetime.timezone.utc).isoformat();R=Path(__file__).resolve().parents[1];U=a.artifact_root.resolve();A=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
# Bind frozen scientific files, not historical Python identity.
old=json.loads((U/'results/context_v2_baseline_20261030/run.json').read_text())['binding']['common']['scientific']
bound={}
for f in ['configurations_sha256','manifests_sha256','core_code_sha256']:bound.update(old[f])
assert len(bound)==19
for rel,h in bound.items():assert sha(U/rel)==h and sha(R/rel)==h,rel
checks['frozen_19_scientific_bindings_equal']=True
copied=[]
for prefix in ['scripts','configs','data/manifests']:
    for path in (U/prefix).rglob('*'):
        if path.is_file():
            rel=path.relative_to(U);assert sha(R/rel)==sha(path),str(rel);copied.append(str(rel))
checks['original_scientific_copies_equal']=len(copied)
for path in (R/'evidence-selected').rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(R/'evidence-selected')
    if path.name=='proof_index.json':
        idx=json.loads(path.read_text());assert sha(U/idx['full_artifact_relative_path'])==idx['full_sha256'];continue
    source=U/rel if path.name!='FINAL_RECEIPT.json' else U.parents[1]/'FINAL_RECEIPT.json'
    assert sha(path)==sha(source),str(rel)
checks['selected_evidence_exact_or_labeled_excerpt']=True
entry_sha=sha(R/'scripts/evaluate_release.py')
for c in ['baseline','early','middle','late','multi','skip']:
    d=A/'verification/new24'/c
    complete=json.loads((d/'complete.json').read_text());binding=json.loads((d/'binding.json').read_text())
    assert complete['passed'] and complete['count']==4 and complete['partial'] and binding['entry_sha256']==entry_sha
checks['new24_same_final_entry_sha']=entry_sha
accept=json.loads((R/'verification/NEW24_ACCEPTANCE.json').read_text());assert accept['passed'] and accept['count']==24
checks['new24_complete_partial_groups']=12
data=json.loads((R/'figures/report_data.json').read_text())
assert len(data['quality'])==12 and sum(x['n'] for x in data['quality'])==3072
assert data['old_measurement_counts']==dict(formal=780,warmup=312)
checks['old_quality_and_performance_scopes_separate']=True
mapping=json.loads((R/'verification/REQUIREMENTS_MAPPING.json').read_text());assert len(mapping)==67
for row in mapping:
    text=(R/row['code_file']).read_text().splitlines();assert row['code_function'] in text[row['line']-1]
checks['requirements_67_actual_source_lines']=True
current_md=[R/'README.md',R/'NOTICE.md',R/'report/REPORT.zh.md',*[R/'docs'/x for x in ['EVIDENCE_INDEX.zh.md','REPRODUCTION.zh.md','REQUIREMENTS_MAPPING.zh.md','PUBLICATION.zh.md','SOURCES_AND_CHANGES.zh.md']],*list((R/'talk').glob('*.md'))]
links=[]
for path in current_md:
    text=path.read_text()
    for url in re.findall(r'\]\(([^)]+)\)',text):
        if re.match(r'^[a-z]+:',url) or url.startswith('#'):continue
        target=(path.parent/url.split('#',1)[0]).resolve();assert target.exists(),(str(path),url);links.append((str(path.relative_to(R)),url))
assert not re.search(r'待作者(?:填写|撰写)|TODO|TBD|example\.com', (R/'report/REPORT.zh.md').read_text())
checks['current_navigation_local_links']=len(links)
checks['history_documents_scope']='v3 history retained separately; old package-relative evidence references require full artifact root'
pdf=PdfReader(R/'report/REPORT.zh.pdf');assert len(pdf.pages)==17
compact=re.sub(r'\s|\u200b','', ''.join(x.extract_text() or '' for x in pdf.pages))
for value in ['3072','54.4922','64.0625','256484928','513028808','7962624']:assert value in compact,value
qa=json.loads((R/'verification/REPORT_RENDER_QA.json').read_text());assert qa['passed'] and qa['pages']==17
for rel,h in qa['outputs'].items():assert sha(R/rel)==h,rel
checks['report_17_pages_text_and_render_binding']=True
talk_pdf=PdfReader(R/'talk/DEFENSE.zh.pdf');assert len(talk_pdf.pages)==12
talk=json.loads((R/'talk/talk_source.json').read_text());assert len(talk)==12 and sum(x['minutes'] for x in talk)==30
with zipfile.ZipFile(R/'talk/DEFENSE.zh.pptx') as z:
    notes=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)];assert len(notes)==12
    for name in notes:assert '目标'.encode() in z.read(name)
checks['talk_12_slides_30_target_minutes_notes_complete']=True
checks['code_map_pdf_one_page']=len(PdfReader(R/'talk/CODE_MAP.zh.pdf').pages)==1
exercise=json.loads((R/'talk/exercise_demo.json').read_text());assert exercise['passed'] and exercise['K']==[768] and exercise['device']=='cpu'
checks['exercise_cpu_only_no_new_quality']=True
# Syntax checks do not execute/mutate the old framework.
for name in ['evaluate_release.py','build_figures.py','build_report.py','build_mapping.py','build_talk_content.py','exercise_demo.py','build_code_sheet.py','validate_delivery.py']:
    ast.parse((R/'scripts'/name).read_text())
blocked=json.loads((A/'publication/ACCESS_BLOCKER.json').read_text());assert blocked['status']=='blocked' and not blocked['uploaded']
checks['publication_claim_honest']=True
tracked=subprocess.check_output(['git','-C',str(R),'ls-files','-z'],text=True).split('\0')
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for rel in tracked:
    if not rel:continue
    path=R/rel
    if not path.exists():continue
    assert path.stat().st_size<50*1024*1024,rel
    if path.suffix.lower() in ['.py','.mjs','.md','.json','.jsonl','.csv','.txt','.sh']:assert not secret.search(path.read_bytes()),rel
checks['git_candidate_no_large_asset_or_credential_signature']=True
result=dict(passed=True,started_at=started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=sys.argv,cwd=str(Path.cwd()),checks=checks,
            not_verified=['public repository/Release/downloads (authentication/capability block)','cold installation/cross-machine','new entry full256','v3 first/stages','Flash/SDPA','user review/independent practice/technical exchange'])
(R/'verification/FINAL_ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
