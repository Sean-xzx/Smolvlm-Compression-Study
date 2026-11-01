# Smolvlm-Compression-Study

Sean-xzx｜SmolVLM-256M-Instruct 无需训练压缩。比较四种 Encoder Top-K 与三层 LLM FFN Skip；TextVQA、ScienceQA-IMG 各固定256题，六配置、12组评估，两方法不叠加。

**[报告 PDF](https://raw.githubusercontent.com/Sean-xzx/Smolvlm-Compression-Study/main/report/REPORT.zh.pdf) · [Markdown](report/REPORT.zh.md) · [Word](https://raw.githubusercontent.com/Sean-xzx/Smolvlm-Compression-Study/main/report/REPORT.zh.docx) · [教授正式题目](docs/ASSIGNMENT.zh.md)**

## 主要结果

context-v2 原3072条预测，每组 n=256。Text 为 MMF 十参考留一软分；Science 为选项准确率（本地标签盲解析，失败保留分母）。

|配置|TextVQA 软分 %|ScienceQA-IMG 准确率 %|
|---|---:|---:|
|Baseline|54.4922|64.0625|
|Early|43.5547|55.0781|
|Middle|42.1484|58.2031|
|Late|47.9688|60.5469|
|Multi|13.6328|32.8125|
|Skip|32.9688|32.4219|

所有变体质量下降，Late在四种Top-K中损失最小。Multi减少59.588%的Encoder主要FLOPs，但质量损失最大；Skip固定16输出token的速度比仅0.993×/1.002×（Text/Science）。

Top-K只缩短Encoder内部序列，postnorm后零回填使LLM视觉位置保持；六配置参数/权重文件均不变，未见明显allocated峰值节省。精确pp、全模型成本及延迟见报告。

## 按题目检查

|检查项|报告答案|实现|证据|
|---|---|---|---|
|模型、数据与评分|[§1 协议](report/REPORT.zh.md#1-实验协议)|[协议](configs/eval_protocol_context_v2.json)、[清单](data/manifests)、[评分](scripts/check_scores.py#L12)|[E1 输入](docs/EVIDENCE_INDEX.zh.md#e1)、[E5 评分](docs/EVIDENCE_INDEX.zh.md#e5)|
|A Top-K四个必答|[§2及附录A–C](report/REPORT.zh.md#2-任务a-encoder-top-k)|[选择/当前FFN/回填](scripts/encoder_topk.py#L153)|[E2 一致性](docs/EVIDENCE_INDEX.zh.md#e2)、[E3 矩阵](docs/EVIDENCE_INDEX.zh.md#e3)|
|B Projector说明|[§3 四类方法与实际模块](report/REPORT.zh.md#3-任务b-projector)|[原Connector适配](scripts/encoder_topk.py#L180)|[E3 形状](docs/EVIDENCE_INDEX.zh.md#e3)|
|C Skip三个必答|[§4 参数/成本/未执行证明](report/REPORT.zh.md#4-任务c-ffn-skip)|[Skip forward](scripts/ffn_skip.py#L16)|[E4 调用/profiler](docs/EVIDENCE_INDEX.zh.md#e4)|
|效率与正确性|[§5–6及附录D](report/REPORT.zh.md#5-实测效率与取舍)|[成本](scripts/analyze_costs.py#L24)、[计时](scripts/benchmark_efficiency.py#L65)|[E6 性能范围](docs/EVIDENCE_INDEX.zh.md#e6)|

[67项详细映射](docs/REQUIREMENTS_MAPPING.zh.md) · [六配置](configs/context_v2) · [结果与图源](figures) · [完整证据索引](docs/EVIDENCE_INDEX.zh.md)

## 复现与范围

按[复现说明](docs/REPRODUCTION.zh.md)执行：clone → 下载[完整科学附件](docs/PUBLICATION.zh.md#下载与版本)并核SHA → 解包 → 准备固定依赖/模型 → 指定artifact-root运行。**单独clone不含完整推理输入。** 可先运行只读结构检查：

```bash
python3 -B scripts/check_repository.py
```

主性能来自cost-perf-v1小样本；v3迁移与新增24条另记，不混入主结果。冷安装、跨机、新入口完整256及Flash/SDPA未验证，Science官方完整评测recipe未复现。[当前main报告与旧Release快照的区别](docs/PUBLICATION.zh.md)已说明。

AI参与适配、实验编排、统计与报告写作；实际贡献和核验见[报告§7.2](report/REPORT.zh.md#72-来源与贡献)。[来源及改动](docs/SOURCES_AND_CHANGES.zh.md) · [许可](NOTICE.md) · [技术交流准备材料](talk/README.zh.md)
