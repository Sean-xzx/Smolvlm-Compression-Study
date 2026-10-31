# Smolvlm-Compression-Study

Sean-xzx（GitHub 账号）｜SmolVLM-256M-Instruct 无需训练压缩实践。

**[报告 PDF](report/REPORT.zh.pdf) · [报告正文/可编辑源](report/REPORT.zh.md) · [教授正式题目](docs/ASSIGNMENT.zh.md) · [关键实现](scripts/encoder_topk.py#L153) · [执行证据](docs/EVIDENCE_INDEX.zh.md) · [接收者复现](docs/REPRODUCTION.zh.md)**

## 任务范围与交付

固定 TextVQA、ScienceQA-IMG 验证集各 256 题，基线、四种 Encoder Top-K 和一种 LLM FFN Skip 共六配置、12 组正式评估。任务 A 保持 LLM 不变，任务 C 保持 Encoder 不变；未训练、扫比例或搜索最优组合。Projector 部分只作题目要求的文字说明。

沿下表检查“要求 → 实现 → 结果 → 证据 → 复现”。需要逐条深入时再查 [67 项详细映射](docs/REQUIREMENTS_MAPPING.zh.md)。

|正式题目分组|实现或配置|回答与结果|证据与运行入口|
|---|---|---|---|
|模型、数据、评分（§1）|[固定协议](configs/eval_protocol_context_v2.json)、[样本清单](data/manifests)、[评分](scripts/check_scores.py#L12)|[报告 §2](report/REPORT.zh.md#2-实验协议与评分)|[E1/E5：输入与评分](docs/EVIDENCE_INDEX.zh.md#e1)；[准备数据](docs/REPRODUCTION.zh.md#4-数据准备与校验)|
|A：Encoder Top-K（§2）|[打分/选集](scripts/encoder_topk.py#L19)、[插入与回填](scripts/encoder_topk.py#L153)、[四配置](configs/context_v2)|[报告 §3–4](report/REPORT.zh.md#3-encoder-top-k实现)、[逐层 Token CSV](figures/layer_tokens.csv)|[E2/E3：一致性与真实少算](docs/EVIDENCE_INDEX.zh.md#e2)；[新评估](docs/REPRODUCTION.zh.md#5-新评估)|
|B：Projector 说明（§3）|[实际连接与恢复网格](scripts/encoder_topk.py#L180)|[报告 §5：四类方法与本模型](report/REPORT.zh.md#5-projector与ffn-skip)|[E3：视觉到 LLM 的形状](docs/EVIDENCE_INDEX.zh.md#e3)，未额外实现四种连接器|
|C：LLM FFN Skip（§4）|[Skip forward](scripts/ffn_skip.py#L16)、[三层配置](configs/context_v2/skip.json)|[报告 §5/8](report/REPORT.zh.md#5-projector与ffn-skip)|[E4：prefill/decode 零调用](docs/EVIDENCE_INDEX.zh.md#e4)；[正确性](docs/REPRODUCTION.zh.md#6-精确重放与正确性)|
|实验与验证（§5）|[计时边界](scripts/benchmark_efficiency.py#L65)、[成本公式](scripts/analyze_costs.py#L24)|[报告 §6–12](report/REPORT.zh.md#6-正式质量结果)、[成本](figures/costs.csv)、[性能](figures/performance.csv)|[E6/E7：版本与测量范围](docs/EVIDENCE_INDEX.zh.md#e6)；[性能/profiler](docs/REPRODUCTION.zh.md#7-性能profiler与成本)|
|提交内容（§6）|[源码](scripts)、[六配置与父 contract](configs/context_v2)、[依赖锁](requirements-morning.lock.txt)|[报告](report)、[图表与图源](figures)|[科学附件及版本](docs/PUBLICATION.zh.md)、[完整原件位置](docs/EVIDENCE_INDEX.zh.md#材料在哪里)|
|来源、AI 与交流（§7）|[来源及改动](docs/SOURCES_AND_CHANGES.zh.md)、[许可](NOTICE.md)|[报告 §17](report/REPORT.zh.md#17-来源与实际贡献)|[slides/讲稿与练习](talk/README.zh.md)，准备材料不证明个人已独立掌握|

## 任务 A：Encoder Top-K

在指定层 **Attention 残差后、当前 FFN 前**，对 head 与有效 query 平均给 key 打分；取最高 K，恢复原序后真实 gather，让当前 FFN 及后层处理缩短序列。层号从 1 开始，K 向上取整；无额外 CLS，各图块独立处理。

|配置|层号|每阶段保留当前序列|每图块实际 K（初始 1024）|
|---|---|---|---|
|Early / Middle / Late|3 / 6 / 9，分别独立运行|50%|512（最终 50%）|
|Multi|3 → 6 → 9，同一次前向|各 50%，在当前序列重新打分|512 → 256 → 128（最终 12.5%）|

Encoder 内部确实缩短；**postnorm 后零回填到原网格，再走原 Connector，因此当前适配没有缩短 LLM 接收的视觉位置**。TextVQA 第 5 题为 13 图块，基线与各变体均为 832 个 LLM 视觉位置；不可把该代表样本数量推广到所有图片。[逐层记录](figures/layer_tokens.csv)与[实际矩阵证据](figures/matrix_evidence.csv)分别展示 Token 与计算形状。

## 任务 B：Projector

Linear/MLP 通常逐 token 映射特征，Q-Former 用学习 query 汇聚，Perceiver Resampler 用固定学习 latent 重采样；这些是概念说明，**不是本仓库额外实现的四组实验**。本模型沿用 HF 的 pixel shuffle + SimpleMLP 单 Linear；剪枝后采用上述零回填适配。完整解释与来源见[报告 §5](report/REPORT.zh.md#5-projector与ffn-skip)。

## 任务 C：FFN Skip

预先选定 LLM 第 **8、16、24** 层同时跳过，支持配置层列表，未搜索最优层。保留 input norm、Attention、Attention 残差与 KV cache；返回 Attention 残差，省去该层 post-attention norm2 和 MLP。三层停用 MLP 参数合计 7,962,624，另有 norm2 参数 1,728；参数仍保存在权重中。[实现](scripts/ffn_skip.py#L16)与[E4 调用/profiler](docs/EVIDENCE_INDEX.zh.md#e4)对应检查。

## 主要结果

原 context-v2 正式结果，每组 256 题；Text 为 MMF 留一软分，Science 为官方索引准确率定义加本地标签盲文本解析，失败保留分母。

|配置|TextVQA 软分 %|ScienceQA-IMG 准确率 %|
|---|---:|---:|
|Baseline|54.4922|64.0625|
|Early|43.5547|55.0781|
|Middle|42.1484|58.2031|
|Late|47.9688|60.5469|
|Multi|13.6328|32.8125|
|Skip|32.9688|32.4219|

所有变体质量均下降，Multi/Skip 损失明显。逐配置 pp 差和实际生成长度见[quality.csv](figures/quality.csv)，本表不混入新增 24 条。

- **参数/存储**：总参数 256,484,928、原权重 513,028,808 bytes 均不变；Skip 是停用而非删权重，没有明显 allocated 峰值节省证据。
- **Token**：Encoder 与 LLM 位置分开看；内部缩短不能冒称端到端视觉 token 同比下降。
- **计算量**：[costs.csv](figures/costs.csv)区分 Encoder/全模型、实际 G/统一 G=16；乘加按 2、主要矩阵/Conv 估算，其他开销另述。
- **实测效率**：[performance.csv](figures/performance.csv)及[每题 median/IQR](figures/performance_samples.csv)来自旧 cost-perf-v1。fixed16 下 Skip 两集约 0.993×/1.002×，收益不明确；自然生成少输出不等于同工作量提速。少 FLOPs 不保证同比 wall 收益，模块独立 median 不可相加减。

详细结果与范围见[报告 §6–13](report/REPORT.zh.md#6-正式质量结果)。Science 64.0625% 未完整复现官方 73.6% recipe；Text normalizer 同源，不冒称异源完整评分器验证。

## 正确性与执行证据

先查[E2 的 100%/空 Skip 一致性](docs/EVIDENCE_INDEX.zh.md#e2)，再查[E3 的真实 50%/Multi 矩阵](docs/EVIDENCE_INDEX.zh.md#e3)和[E4 的 Skip 零调用](docs/EVIDENCE_INDEX.zh.md#e4)。100% bypass 不代替真实剪枝路径证明；有限样本 hook/profiler 不担保任意输入。

Git 只放精选 JSON、CSV 和 B2 导航摘录；完整 3072 原子结果、原输入图及 trace 通过[已发布科学包](docs/PUBLICATION.zh.md#下载与版本)取得。原机器绝对路径是历史执行信息；阅读用本仓库相对链接，运行用显式 artifact-root。

## 复现、限制与来源

按[接收者说明](docs/REPRODUCTION.zh.md)执行：clone 阅读 → 下载指定科学附件 → 核 SHA → 解包 → 固定依赖/模型 → 传 artifact-root 运行。**单独 clone 不含完整推理输入。** 可先用 `python3 -B scripts/check_repository.py` 检查阅读材料；这是只读结构检查，不是科学正确性或 GPU 就绪验收。

旧 3072 质量、旧 780 正式/312 预热、v3 迁移 120/48 与新增 24 条分别登记。新增 24 条仅为同机热缓存薄入口验证；冷安装、跨机、新入口完整256、Flash/SDPA 和 v3 first/stages 仍未验证。本次只整理仓库，未追加实验；报告和 slides 保持原字节。

当前发布状态统一见[PUBLICATION](docs/PUBLICATION.zh.md)。Release 固定快照与当前 main 并非字节一致；上传时状态与旧验收的原结论保留。来源许可见[NOTICE](NOTICE.md)、[上游及改动](docs/SOURCES_AND_CHANGES.zh.md)。AI 主要完成适配、编排、自动核验、正文/图表和本次导航整理；未宣称 Sean-xzx 已亲写、审阅或独立掌握。教授指定的现场环节需本人独立完成、不调用 AI/Agent。
