# 证据导航

科学权威版本为原postday3-v3完整包；本仓库 `evidence-selected` 包含字节一致的精选JSON及明确标注的B2导航摘录，未含全部raw/trace/图片。下文U表示解包 `artifacts/Shen-Wu-Challenge`，E编号与报告一致。不要把历史passed当本轮重跑。

|索引|结论与直接证据|完整包内路径与范围|
|---|---|---|
|E1 质量与固定输入|[本轮只读质量汇总](../figures/quality.csv)、[协议](../configs/eval_protocol_context_v2.json)、[两清单](../data/manifests)|U/results/context_v2_{配置}_20261030/{数据集}/samples/row_*.json；12×256=3072；原始prompt/IDs/raw/score/parse/T/G。不是新推理|
|E2 数值与一致性|[B2 proof索引与原SHA](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b2/proof_index.json)|U/evidence/day2_afternoon/20261030_context_v2/b2；100%8路径、空Skip、真实50%/Multi、mask/残差/选集/恢复；有限样本|
|E3 真少算与Token|[B4总验收](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4/report.json)、[Multi形状/矩阵/时间](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4/multi.json)、[精简矩阵表](../figures/matrix_evidence.csv)、[逐层Na/Nf](../figures/layer_tokens.csv)|U/evidence/day2_afternoon/20261030_context_v2/b4/{baseline,early,middle,late,multi}.json与_trace.json；b6/encoder_layer_lengths.csv共24576；Text5 B13/T889|
|E4 Skip未执行|[基线调用](../evidence-selected/evidence/day2_evening/cost_perf_v1/skip_trace/baseline/skip_trace_proof.json)、[Skip调用](../evidence-selected/evidence/day2_evening/cost_perf_v1/skip_trace/skip/skip_trace_proof.json)|U/evidence/day2_evening/cost_perf_v1/skip_trace；prefill+1decode，目标norm2/MLP0，gate/up/down marker缺席，Attention保留；实际trace在完整包|
|E5 评分与失败案例|[评分摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/summary.json)、[原始案例](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/failure_cases.json)、[评分来源](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/sources/provenance.json)|U/evidence/day2_evening/cost_perf_v1/scoring；Text1536闭式聚合核对，同源normalizer；Science标签盲解析，失败保留分母|
|E6 成本与主性能|[成本原摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/costs/summary.json)、[旧性能摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/performance_summary.json)、[重新汇总性能](../figures/performance.csv)、[每题median/IQR](../figures/performance_samples.csv)、[成本表](../figures/costs.csv)|U/evidence/day2_evening/cost_perf_v1/costs/{costs_actual_G,costs_G16}.csv各3072；{natural,first,fixed16,stages}/{配置}/measurements.jsonl，780正式312预热；锚点另计|
|E7 独立解包v3|[原回执](../evidence-selected/FINAL_RECEIPT.json)、[正确性](../evidence-selected/validation/postday3_v3/checks/correctness/correctness.json)、[短profiler](../evidence-selected/validation/postday3_v3/checks/profiler/profiler.json)、[v3性能](../evidence-selected/validation/postday3_v3/performance_summary.json)|U/validation/postday3_v3；27命令，性能120正式48预热，锚点20/8；同机共享解释器/热缓存；不与旧780拼毫秒|
|E8 本轮24条|[验收](../verification/NEW24_ACCEPTANCE.json)、[精确argv/cwd/起止/退出](../verification/commands.json)、[新入口](../scripts/evaluate_release.py)|本轮verification/new24/{配置}；每配置两集前2题，真实24预测，各集partial；完整新输出作为独立附件候选提供，不入旧主表|

所有图表的轻量CSV/JSON、[输入SHA索引](../figures/source_inputs.json)、[figure_manifest](../figures/figure_manifest.json)、[绘图脚本](../scripts/build_figures.py)均可查。数值来自真实文件，不手填曲线。E8新增文件使用本轮清单；原v3回执不担保新增代码或文稿。

复现先读[接收者命令页](REPRODUCTION.zh.md)。已保存的历史检查与本轮实际执行范围见报告§14/15。任何完整trace或3072逐题细节都从完整包定位，轻量Git不假装包含所有父依赖。
