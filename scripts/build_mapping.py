"""Produce the 67-row final requirement map from the read-only audit matrix."""
from pathlib import Path
import argparse,json,re
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1]
spec={
'D1':('1.1','run_original.py:load_original_model','E1','固定模型revision；新24实测'),
'D2':('1.1','evaluate_context.py:manifests_for','E1','固定两数据集revision'),
'D3':('1.1/5.1','evaluate_context.py:manifests_for','E1','旧12×256，只读汇总；不重复推理'),
'D4':('1.1','inspect_data.py:prepare_manifests','E1','seed/split/ID/meta已保存；HF重建本轮未跑'),
'D5':('1.1','context_runtime.py:load_context_model','E1/E8','原Processor与输入shape；新24输入对齐'),
'D6':('1.1','context_runtime.py:build_question','E1/E8','hint合法，answer/lecture/solution不输入'),
'D7':('1.2/5.1','check_scores.py:textvqa_score','E5','Text留一软分；同源normalizer'),
'D8':('1.2/5.1','check_scores.py:extract_choice','E5','官方索引准确率定义，本地标签盲解析'),
'A1':('2.1','encoder_topk.py:EncoderTopKPatch','E2','原权重无需训练'),
'A2':('2.1','encoder_topk.py:__init__','E2','层列表/阶段率检查；演练另版配置'),
'A3':('2.1 图1','encoder_topk.py:prune_layer','E2/E3','Attention残差后、当前FFN前'),
'A4':('2.1','encoder_topk.py:attention_scores','E2','FP32 head/有效query均值给key'),
'A5':('2.1','encoder_topk.py:select_tokens','E2','稳定TopK、恢复原序、真实gather'),
'A6':('2.2/2.4 附录A/C','encoder_topk.py:prune_layer','E3','当前Nf及后层Na真实缩短'),
'A7':('2.1 配置表','experiment_runtime.py:patch_for','E1/E3','独立3/6/9层各50%'),
'A8':('2.1/2.4 附录C','encoder_topk.py:prune_layer','E2/E3','同前向逐阶段重打分512/256/128'),
'A9':('2.1','encoder_topk.py:select_tokens','E2','1起层号、ceilK；有效[16,5]受控例'),
'A10':('2.1/2.5','encoder_topk.py:restore_grid','E2/E3','无额外CLS；块独立、postnorm后回填'),
'A11':('2.2 附录A','encoder_topk.py:connect','E1/E3','参数/文件不变；完整12层Na/Nf'),
'A12':('2.3/5.2 附录B','analyze_costs.py:components','E6','Encoder及全模型，实际G/G16分开'),
'A13':('2.3 附录B3','encoder_topk.py:select_tokens','E6','score/cast/排序/移动/同步单列范围'),
'A14':('2.4 附录C','profile_context.py:main','E3','Text5 prehook当前/下一层'),
'A15':('2.4 附录C1/C2','release_profile.py:encoder_trace','E3/E7','Multi三阶段shape与CPU/CUDA时间'),
'A16':('2.5 图1','encoder_topk.py:connect','E3','Text5 13块→832视觉位置保持'),
'B1':('3','encoder_topk.py:connect','E3','Linear pointwise通常保持N；概念说明'),
'B2':('3','encoder_topk.py:connect','E3','MLP pointwise通常保持N；未额外实现'),
'B3':('3','encoder_topk.py:connect','E3','BLIP-2学习query汇聚；仅概念来源'),
'B4':('3','encoder_topk.py:connect','E3','Flamingo视觉固定学习latent重采样；仅说明'),
'B5':('3','encoder_topk.py:connect','E3','原HF SimpleMLP单Linear；pixel shuffle改N'),
'B6':('3/2.5','encoder_topk.py:restore_grid','E2/E3','postnorm后零网格，原Connector不改'),
'C1':('4.1','ffn_skip.py:skip_forward','E4','三层8/16/24同时Skip'),
'C2':('4.1','ffn_skip.py:__init__','E4','可配置列表/范围/重复检查'),
'C3':('4.1','ffn_skip.py:skip_forward','E2/E4','保Attention/inputnorm/cache；省norm2/MLP'),
'C4':('4.1','ffn_skip.py:skip_forward','E1','预先分散，未宣称最优'),
'C5':('4.2','run_original.py:load_original_model','E4','MLP7962624、norm2另1728；权重未删'),
'C6':('4.3 附录B','analyze_costs.py:components','E6','15925248(T+G−1)，prefill+G−1decode'),
'C7':('4.4','benchmark_efficiency.py:skip_trace','E4/E7','零调用+投影marker缺席；v3 180→162'),
'V1':('5.1','experiment_runtime.py:patch_for','E1','六配置两集3072；不叠加方法'),
'V2':('6','release_checks.py:correctness','E2/E7','8路径32张量，score/scatter0；不代替50%'),
'V3':('6','verify_skip.py:empty_stage','E2/E7','空Skip独立cache prefill/decode与IDs'),
'V4':('6','verify_encoder_topk.py:equal','E2/E7','maxerr0，atol/rtol0；Encoder未独立decode logits'),
'V5':('2.4/4.4 附录C','profile_context.py:main','E3/E4','配置/ID/模块明确；诊断不代替稳定延迟'),
'V6':('1.1/5.3','benchmark_efficiency.py:generation_block','E6','同硬件BF16/batch1/eager；不混版本'),
'V7':('5.3 附录D2/D3','benchmark_efficiency.py:timed','E6','旧780/312；n/G/边界/同步/重复明确'),
'V8':('1.1/2.1','encoder_topk.py:prune_layer','E2','显式eager权重；Flash/SDPA未测'),
'V9':('2.3/4.3 附录B','analyze_costs.py:components','E6','主要矩阵Conv乘加2，辅助范围明示'),
'V10':('5.3/7.1','benchmark_efficiency.py:stage_block','E6','自然G/选择回填开销；独立median不相加减'),
'V11':('5.1/7.1','check_scores.py:score_response','E1/E6','保留Multi/Skip负结果与无存储收益'),
'S1':('7.3/7.4','evaluate_release.py:main','E7/E8','公开main与固定Release已发布；当前状态见PUBLICATION；两者版本不同'),
'S2':('1/2.1/4.1','experiment_runtime.py:patch_for','E1/E8','源码六配置/依赖锁/版本齐全'),
'S3':('7.3','inspect_data.py:prepare_manifests','E1/E7','两数据路径职责明确；冷HF重建未测'),
'S4':('7.3','evaluate_release.py:main','E8','新24预测与聚合已测；limit256未跑'),
'S5':('7.3 附录D3','release_checks.py:main','E7','正确性/性能/profiler/CPU命令；v3 first/stages未跑'),
'S6':('0至7 附录A至E','build_report.py:render_document','E1至E8','Markdown为母稿、Word/PDF同源；新渲染见REPORT_REFACTOR_QA，旧回执保持历史'),
'S7':('2.4/4.4/6 附录C','benchmark_efficiency.py:skip_trace','E2/E3/E4/E7','精选proof可点；完整trace在科学包'),
'S8':('2.1 配置表','experiment_runtime.py:config_path','E1','层/阶段率/K/最终率/Skip表'),
'S9':('5.1 图3','build_figures.py:read','E1','两指标与同版pp；解析失败保留分母'),
'S10':('2.2/4.2','run_original.py:load_original_model','E4','总参数/停用参数/文件大小'),
'S11':('2.2/2.5 附录A','encoder_topk.py:connect','E3','逐层Na/Nf与Projector/LLM形状'),
'S12':('5.2/5.3 附录B/D','analyze_costs.py:components','E6','Encoder/全模型FLOPs、延迟/G/同版比'),
'S13':('5.1 附录D','build_figures.py:csvread','E1/E6','Text5案例、全256实际G/成本范围、小n性能统计'),
'T1':('7.2/7.4','encoder_topk.py:EncoderTopKPatch','E5','来源/license/固定commit与必要适配'),
'T2':('7.2','evaluate_release.py:main','E8','AI实际用途与自动检查明确；不冒称人工'),
'T3':('7.2','context_runtime.py:validate_protocol','E5/E7','历史勘误/失败保留；不发布完整聊天'),
'T4':('7.2','evaluate_release.py:main','E8','个人理解/责任不能由AI文件证明；尚未验收'),
'T5':('7.2','encoder_topk.py:select_tokens','E8','30分钟slides/讲稿/代码/练习是准备材料；交流未发生'),
'T6':('7.2','encoder_topk.py:select_tokens','E8','离线索引齐全；现场指定不用AI需用户未来执行'),
}
rows=[]
for line in a.audit.read_text().splitlines():
    m=re.match(r'^\|([DABCVST]\d+)\|([^|]+)\|',line)
    if m:
        ident,requirement=m.groups()
        if ident=='A14':requirement='pre-hook证明被丢弃token未进入当前FFN和后层'
        section,code,evidence,scope=spec[ident];file,func=code.split(':');text=(R/'scripts'/file).read_text();matches=list(re.finditer(r'^\s*(?:def|class) '+re.escape(func)+r'\b',text,re.M))
        assert matches,(ident,code)
        line_no=text[:matches[0].start()].count('\n')+1
        # For indentation matches the regex may consume a preceding newline.
        while not text.splitlines()[line_no-1].strip().startswith(('def ','class ')):line_no+=1
        status='not_assessed' if ident=='T4' else 'prepared_future_event_unverified' if ident in ['T5','T6'] else 'published_with_scope' if ident=='S1' else 'covered_with_scope'
        rows.append(dict(id=ident,requirement=requirement,report_section=section,code_file='scripts/'+file,code_function=func,line=line_no,evidence=evidence,verified_scope=scope,status=status))
assert len(rows)==67 and set(spec)=={r['id'] for r in rows},(len(rows),set(spec)-{r['id'] for r in rows})
header='# 教授原文67项最终映射\n\n报告professor-report-v2以节号及附录定位，七个必答见§2.2–2.5与§4.2–4.4，完整原文索引在报告附录E；代码行号从本轮实际文件读取。E编号见[证据索引](EVIDENCE_INDEX.zh.md)。原文[原字节副本](ASSIGNMENT.zh.md)。covered_with_scope表示有实现/已保存实验及本轮只读核对，不表示本轮重跑全部历史实验。个人独立能力、未来交流与公开发布分别标记，不以自动检查替代。S1按[当前发布状态](PUBLICATION.zh.md)更新；历史验收原结论保留。\n\n|编号|原文要求|报告节/图表|实际代码及行|证据/命令|范围与未验证边界|状态|\n|---|---|---|---|---|---|---|\n'
for r in rows:
    header+=f"|{r['id']}|{r['requirement']}|§{r['report_section']}|[{r['code_function']}](../{r['code_file']}#L{r['line']})|{r['evidence']}；见REPRODUCTION命令|{r['verified_scope']}|{r['status']}|\n"
(R/'docs/REQUIREMENTS_MAPPING.zh.md').write_text(header)
(R/'verification/REQUIREMENTS_MAPPING.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print('67 requirements mapped with actual source lines')
