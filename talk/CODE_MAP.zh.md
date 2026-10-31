# 代码定位一页速查

行号从最终源文件AST读取；冻结科学文件字节与v3相同。轻量仓库的原loader需要完整artifact-root，不能直接偷读作者工程。

|用途|文件和函数|实际行号|
|---|---|---|
|head/query→key评分|[scripts/encoder_topk.py](../scripts/encoder_topk.py#L19) `attention_scores`|19|
|ceilK/稳定选集/原序/gather|[scripts/encoder_topk.py](../scripts/encoder_topk.py#L34) `select_tokens`|34|
|两个残差与当前FFN|[scripts/encoder_topk.py](../scripts/encoder_topk.py#L153) `prune_layer`|153|
|postnorm后恢复网格/原Connector|[scripts/encoder_topk.py](../scripts/encoder_topk.py#L180) `connect`|180|
|Attention/cache保留，省norm2+MLP|[scripts/ffn_skip.py](../scripts/ffn_skip.py#L16) `skip_forward`|16|
|只传合法问题字段/hint|[scripts/context_runtime.py](../scripts/context_runtime.py#L47) `build_question`|47|
|标签盲Science解析|[scripts/check_scores.py](../scripts/check_scores.py#L18) `extract_choice`|18|
|MMF留一软分|[scripts/check_scores.py](../scripts/check_scores.py#L12) `textvqa_score`|12|
|接收者新预测/显式artifact-root|[scripts/evaluate_release.py](../scripts/evaluate_release.py#L18) `main`|18|
|复用汇总与partial/请求分母|[scripts/evaluate_release.py](../scripts/evaluate_release.py#L59) `aggregate`|59|
|严格历史重放|[scripts/replay_release.py](../scripts/replay_release.py#L74) `main`|74|
|同步/事件/峰值计时边界|[scripts/benchmark_efficiency.py](../scripts/benchmark_efficiency.py#L65) `timed`|65|
|预热/重复/实际G|[scripts/benchmark_efficiency.py](../scripts/benchmark_efficiency.py#L80) `generation_block`|80|
|Encoder与全模型成本|[scripts/analyze_costs.py](../scripts/analyze_costs.py#L24) `components`|24|

协议：configs/eval_protocol_context_v2.json。两manifest：data/manifests。预先冻结科学层/率与Skip列表：configs/context_v2。报告§8有公式，E1–E8索引在docs/EVIDENCE_INDEX.zh.md。
