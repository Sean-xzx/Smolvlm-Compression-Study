# 离线速查

现场指定环节不用AI/Agent。可以查本仓库代码、报告、完整证据包和本页；用户独立演练尚未进行。

## 两条计算式

Encoder层：8BNa dv² + 4BNa²dv + 4BNf dv mv。dv=768、mv=3072；Attention用Na，当前FFN用Nf。

Skip减少：15925248(T+G−1)主要矩阵FLOPs。prefill q=s=T，decode共G−1步q=1/s=T+j；首输出属于prefill。乘加2。权重仍保留。

## 关键形状

Text5：13独立块，每块1024。Multi 512/256/128；postnorm后回填1024；pixel shuffle 64×12288；Linear到576；LLM视觉832、总T889。第6层Multi fc1 [13,256,768]，矩阵[3328,768]×[768,3072]。

## 入口职责

1. docs/REPRODUCTION.zh.md先完成环境/固定模型缓存/原包SHA校验。
2. release_checks.py --mode data 是就绪校验，不是下载器。
3. scripts/evaluate_release.py --artifact-root U --configuration early --limit 2 --output-dir NEW 是新预测。
4. U/scripts/replay_release.py --configuration early --limit 1 --output-dir NEW 是强制历史相等重放。
5. U/scripts/release_checks.py --mode correctness/performance/profiler；性能另需configuration和phase。
6. U/scripts/analyze_costs.py --probe-root U/evidence/day2_evening/cost_perf_v1 --output-dir NEW 只离线算成本。

所有NEW为空、位于U之外。不得python -O，不能改旧hash过门禁。只改练习副本，不在旧结果上写输出。

## 证据定位

E1质量3072、E2数值proof、E3短张量/矩阵、E4 Skip零调用、E5评分/失败、E6旧780性能与成本、E7 v3独立解包、E8新24条。完整路径在docs/EVIDENCE_INDEX.zh.md。

汇报数字不混用：质量每组256；旧自然/first每集4题、fixed/stages每集1题、2预热5正式，共780/312；v3 120/48；本轮新24条partial。5次重复不是5个独立任务。

代码函数行号见CODE_MAP.zh.md；练习答案见EXERCISES_AND_SOLUTIONS.zh.md；未知结果不推测成实测。
