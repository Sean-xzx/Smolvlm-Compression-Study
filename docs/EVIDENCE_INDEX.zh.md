# 按正式题目检查证据

从[首页检查表](../README.md)进入；E1–E8 编号与新版报告/67项映射保持一致。这里只整理已保存的证据，不新增实验，也不把历史 `passed` 当作此次重跑。

## 材料在哪里

- **Git 精选**：`evidence-selected/` 的 JSON、`figures/` 的 CSV/JSON，以及本页链接。B2 `proof_index.json` 明确是导航摘录，不含完整张量/IDs，不可当完整证明。
- **完整科学原件**：[postday3-v3 科学附件](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/Shen-Wu-Challenge-postday3-v3.tar.gz)。下文 **U** 为解包后的 `artifacts/Shen-Wu-Challenge`；表中 U 路径均相对该根。包内旧绝对路径仅作历史溯源。
- **新增预测原件**：[已发布 new24 验证附件](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/new24-verification-20261031.tar.gz)，与旧科学包分开；包内前缀为 `new24/`。
- 阅读无需模型/GPU；运行与包校验顺序见[复现页](REPRODUCTION.zh.md)，附件/hash/历史状态见[当前发布页](PUBLICATION.zh.md)。

## E1

**固定输入、六配置质量与参数/Token 结果（原文 §1、A 必答1/4、§6）。**

|证明对象与配置/样本|Git 直接阅读|完整 U 相对位置|边界|
|---|---|---|---|
|六配置×两集各256，原3072正式质量|[quality.csv](../figures/quality.csv)、[协议](../configs/eval_protocol_context_v2.json)、[清单/meta](../data/manifests)|`results/context_v2_{配置}_20261030/{数据集}/samples/row_*.json` 及各 `run.json`；配置名为 baseline/early/middle/late/multi/skip|保存 prompt/IDs/raw/score/parse/T/G；CSV为旧记录只读汇总，不是新推理|
|逐层 Na/Nf 与图块→Connector→LLM|[layer_tokens.csv](../figures/layer_tokens.csv)、[report_data.json](../figures/report_data.json)|`evidence/day2_afternoon/20261030_context_v2/b6/encoder_layer_lengths.csv`、上述逐题记录|Text5 的832视觉位置是代表案例；零回填保持原Connector/LLM位置，不代表所有图像相同|
|参数与权重存储|[报告 §2.2/4.2](../report/REPORT.zh.md#3-任务b-projector)、[原包回执](../evidence-selected/FINAL_RECEIPT.json)|模型配置/源绑定及各原始 run/runtime 记录，实际模型权重按固定HF revision取得|参数/权重文件不变；停用参数不等于删除；Git不分发模型权重|

## E2

**100% bypass、空 Skip 与真实 50% 路径（原文 §5）。**

|配置/样本与证明对象|Git 直接阅读|完整 U 相对位置|比较对象与边界|
|---|---|---|---|
|Early/Middle/Late单点100%、Multi每阶段100%；Text5/Science2|[v3 correctness.json](../evidence-selected/validation/postday3_v3/checks/correctness/correctness.json) 的 `results.equivalence`|`validation/postday3_v3/checks/correctness/correctness.json`|8路径：Encoder/postnorm/Connector/prefill logits与生成IDs；保存 maxerr0/atol0/rtol0，score/scatter关闭；未独立比较Encoder decode logits|
|空Skip；两集固定首题|同文件的 `results.skip`|`validation/postday3_v3/checks/correctness/`，以及 `evidence/day2_afternoon/20261030_context_v2/b2/proof.json`|含独立cache一步decode与完整IDs；不代替非空Skip执行证明|
|真实50%及Multi、mask/选集/残差/回填、受控fixture|[B2索引与原SHA](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b2/proof_index.json)|`evidence/day2_afternoon/20261030_context_v2/b2/proof.json` 的 `pruned/controlled/toys/restoration` 等；同目录 `controlled_*.pt`|索引不是完整证明；实际张量/IDs须查约50MB原proof；受控fixture与真实样本分开|

不要把100%路径输出一致，解释为真实50%路径没有丢弃计算；后者看 E3 的 hook 和矩阵。

## E3

**被丢弃 token 未进入当前 FFN/后层，及压缩如何传到 LLM（A 必答2/3/4、B）。**

- 配置：baseline、early、middle、late、multi；代表样本 TextVQA row5，13图块，输入 T=889。Multi 三次在当前序列重新打分并剪到512/256/128。
- 直接查：[B4总报告](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4/report.json)、[Multi明细](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4/multi.json)、[矩阵表](../figures/matrix_evidence.csv)。各配置明细均在[同一精选目录](../evidence-selected/evidence/day2_afternoon/20261030_context_v2/b4)。检查当前FFN及下一层Attention的pre-hook输入、矩阵shape与CPU/CUDA时间；不是只看最终输出形状。
- 完整原件：U的 `evidence/day2_afternoon/20261030_context_v2/b4/{baseline,early,middle,late,multi}.json` 和对应 `*_trace.json`；v3再次诊断见 `validation/postday3_v3/checks/profiler/multi/` 及[profiler摘要](../evidence-selected/validation/postday3_v3/checks/profiler/profiler.json)。
- postnorm后回填保留原网格，Connector输入/输出和最终LLM视觉位置见[报告 §2.5/3](../report/REPORT.zh.md#25-必答a4-从视觉输入到llm的传递)与上述JSON。Projector四类是文字说明，未增加四种连接器实验。
- 边界：这些是固定诊断样本；嵌套CPU/device事件有重叠，不能当稳定延迟基准或相加计算加速比。

## E4

**三层 FFN 确实未执行，覆盖 prefill + 一步 decode（C 必答3）。**

|配置/样本|Git 直接阅读|完整 U 相对位置|证明边界|
|---|---|---|---|
|baseline/skip，TextVQA row1848，13图块，T=888；Skip8/16/24|[基线调用](../evidence-selected/evidence/day2_evening/cost_perf_v1/skip_trace/baseline/skip_trace_proof.json)、[Skip零调用/marker](../evidence-selected/evidence/day2_evening/cost_perf_v1/skip_trace/skip/skip_trace_proof.json)|`evidence/day2_evening/cost_perf_v1/skip_trace/{baseline,skip}/` 的proof与trace|prefill及decode目标norm2/MLP为0、gate/up/down marker缺席；Attention/inputnorm与cache保留；不是每个生成步均录trace|
|v3迁移诊断，TextVQA row5|[profiler摘要](../evidence-selected/validation/postday3_v3/checks/profiler/profiler.json)|`validation/postday3_v3/checks/profiler/{baseline,skip}/`|独立诊断，不能混入旧性能毫秒表|

停用参数统计和理论节省公式见[报告 §4.2–4.3](../report/REPORT.zh.md#43-必答c2-prefill与decode成本)、[components](../scripts/analyze_costs.py#L24)；参数仍存储，norm2与MLP参数分列。

## E5

**官方评分口径与失败分母（原文 §1）。**

[评分摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/summary.json)、[失败案例](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/failure_cases.json)、[来源](../evidence-selected/evidence/day2_evening/cost_perf_v1/scoring/sources/provenance.json)对应 U 的 `evidence/day2_evening/cost_perf_v1/scoring/`。Text1536输出作留一软分聚合核对，normalizer仍同源；Science为官方索引准确率定义加本地标签盲解析，失败记0并保留分母，lecture/solution不作输入。实现见[check_scores.py](../scripts/check_scores.py#L12)，不冒称异源完整评分器。

## E6

**Encoder/全模型成本与规范性能（A 必答2、C 必答2、原文 §5）。**

|量与样本范围|Git 结果入口|完整 U 相对位置|统计边界|
|---|---|---|---|
|实际G/统一G=16，两数据集全部固定256题|[costs.csv](../figures/costs.csv)、[原摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/costs/summary.json)、[逐层Token](../figures/layer_tokens.csv)|`evidence/day2_evening/cost_perf_v1/costs/{costs_actual_G,costs_G16}.csv` 各3072行，`evidence/day2_afternoon/20261030_context_v2/b6/encoder_layer_lengths.csv`|主要矩阵/Conv、乘加2；Encoder/prefill/decode分项；打分/排序/回填等额外范围见报告附录B3，不称硬件FLOPs实测|
|旧cost-perf-v1，六配置：natural/first每集4题，fixed16/stages每集1题，各2预热5正式|[performance.csv](../figures/performance.csv)、[每题median/IQR](../figures/performance_samples.csv)、[原摘要](../evidence-selected/evidence/day2_evening/cost_perf_v1/performance_summary.json)|`evidence/day2_evening/cost_perf_v1/{natural,first,fixed16,stages}/{配置}/measurements.jsonl`|780正式/312预热，锚点另计；同硬件BF16/batch1/eager，同步包围native generate；不含加载/Processor/H2D/评分/I/O，生产选择/回填保留|

公式在[报告附录B](../report/REPORT.zh.md#附录b-flops推导与分项)，计时边界在[报告 §5.3](../report/REPORT.zh.md#53-固定长度与自然生成延迟)和[benchmark_efficiency.py](../scripts/benchmark_efficiency.py#L65)。自然生成G变化不能替代fixed16比较；独立模块median不可相加减推因果，少FLOPs不保证同比wall/存储/峰值收益。

## E7

**独立解包 v3 的迁移验证，区别于旧780主性能。**

[原包回执](../evidence-selected/FINAL_RECEIPT.json)、[v3正确性](../evidence-selected/validation/postday3_v3/checks/correctness/correctness.json)、[v3 profiler](../evidence-selected/validation/postday3_v3/checks/profiler/profiler.json)、[v3性能摘要](../evidence-selected/validation/postday3_v3/performance_summary.json)对应 U 的 `validation/postday3_v3/`。原保存27命令，性能120正式/48预热、锚点20/8；同机共享解释器/HF热缓存，非冷安装/跨机。v3 first/stages 未实跑；图表的旧780与此版本不拼接毫秒。

## E8

**新增薄入口真实24条，只证明新评估小样本链路（非新增主实验）。**

[NEW24_ACCEPTANCE](../verification/NEW24_ACCEPTANCE.json)、[原始精确命令](../verification/commands.json)、[evaluate_release.py](../scripts/evaluate_release.py#L18)。六配置各处理Text索引5/19和Science2/9，各数据集均partial，24条真实generate、无异常/解析失败；历史相等仅作诊断。完整输出已在[独立验证附件](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/new24-verification-20261031.tar.gz)发布，成员前缀 `new24/{配置}/`。旧3072表不添加这些数据；完整256经新入口仍未验证。

## 阅读与运行分开

[复现页](REPRODUCTION.zh.md)给出可操作的相对命令与显式artifact-root；精确执行回执保留原机器argv/cwd，不直接复制其绝对路径到另一机器。只读结构检查不加载模型，不重新验证上述科学命题。历史与当前发布状态统一在[PUBLICATION](PUBLICATION.zh.md)。
