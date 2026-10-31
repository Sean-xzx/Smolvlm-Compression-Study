# 教授原文67项最终映射

报告以节号定位，图/表在相应节内；代码行号从本轮实际文件读取。E编号见[证据索引](EVIDENCE_INDEX.zh.md)。原文[原字节副本](ASSIGNMENT.zh.md)。covered_with_scope表示有实现/已保存实验及本轮只读核对，不表示本轮重跑全部历史实验。个人独立能力、未来交流与公开发布分别标记，不以自动检查替代。S1按[当前发布状态](PUBLICATION.zh.md)更新；历史验收原结论保留。

|编号|原文要求|报告节/图表|实际代码及行|证据/命令|范围与未验证边界|状态|
|---|---|---|---|---|---|---|
|D1|指定SmolVLM-256M-Instruct|§2|[load_original_model](../scripts/run_original.py#L166)|E1；见REPRODUCTION命令|固定模型revision；新24实测|covered_with_scope|
|D2|指定TextVQA与ScienceQA-IMG|§2|[manifests_for](../scripts/evaluate_context.py#L24)|E1；见REPRODUCTION命令|固定两数据集revision|covered_with_scope|
|D3|每集固定256，所有配置同题|§2/6|[manifests_for](../scripts/evaluate_context.py#L24)|E1；见REPRODUCTION命令|旧12×256，只读汇总；不重复推理|covered_with_scope|
|D4|保存数据版本、划分、种子与样本ID|§2|[prepare_manifests](../scripts/inspect_data.py#L71)|E1；见REPRODUCTION命令|seed/split/ID/meta已保存；HF重建本轮未跑|covered_with_scope|
|D5|固定图像预处理|§2|[load_context_model](../scripts/context_runtime.py#L63)|E1/E8；见REPRODUCTION命令|原Processor与输入shape；新24输入对齐|covered_with_scope|
|D6|固定提示词与生成参数|§2|[build_question](../scripts/context_runtime.py#L47)|E1/E8；见REPRODUCTION命令|hint合法，answer/lecture/solution不输入|covered_with_scope|
|D7|Text官方评分口径及所用实现|§2/6|[textvqa_score](../scripts/check_scores.py#L12)|E5；见REPRODUCTION命令|Text留一软分；同源normalizer|covered_with_scope|
|D8|Science官方评分口径及所用实现|§2/6|[extract_choice](../scripts/check_scores.py#L18)|E5；见REPRODUCTION命令|官方索引准确率定义，本地标签盲解析|covered_with_scope|
|A1|无需训练|§1/3|[EncoderTopKPatch](../scripts/encoder_topk.py#L75)|E2；见REPRODUCTION命令|原权重无需训练|covered_with_scope|
|A2|支持剪枝层列表和各阶段比例|§3|[__init__](../scripts/encoder_topk.py#L76)|E2；见REPRODUCTION命令|层列表/阶段率检查；演练另版配置|covered_with_scope|
|A3|Attention残差后、当前FFN前插入|§3 图1|[prune_layer](../scripts/encoder_topk.py#L153)|E2/E3；见REPRODUCTION命令|Attention残差后、当前FFN前|covered_with_scope|
|A4|按head/query平均给key打分|§3|[attention_scores](../scripts/encoder_topk.py#L19)|E2；见REPRODUCTION命令|FP32 head/有效query均值给key|covered_with_scope|
|A5|选最高K并恢复原序|§3|[select_tokens](../scripts/encoder_topk.py#L34)|E2；见REPRODUCTION命令|稳定TopK、恢复原序、真实gather|covered_with_scope|
|A6|当前FFN及后层真正变短|§4/12 图2|[prune_layer](../scripts/encoder_topk.py#L153)|E3；见REPRODUCTION命令|当前Nf及后层Na真实缩短|covered_with_scope|
|A7|Early/Middle/Late分别独立50%|§3 配置表|[patch_for](../scripts/experiment_runtime.py#L63)|E1/E3；见REPRODUCTION命令|独立3/6/9层各50%|covered_with_scope|
|A8|Multi同一前向依次每阶段保留当前50%|§3/4/12|[prune_layer](../scripts/encoder_topk.py#L153)|E2/E3；见REPRODUCTION命令|同前向逐阶段重打分512/256/128|covered_with_scope|
|A9|说明层号和K取整|§3|[select_tokens](../scripts/encoder_topk.py#L34)|E2；见REPRODUCTION命令|1起层号、ceilK；有效[16,5]受控例|covered_with_scope|
|A10|特殊token/图块/后续适配|§3/5|[restore_grid](../scripts/encoder_topk.py#L62)|E2/E3；见REPRODUCTION命令|无额外CLS；块独立、postnorm后回填|covered_with_scope|
|A11|参数量、权重文件大小、逐层token及指标区别|§4/5|[connect](../scripts/encoder_topk.py#L180)|E1/E3；见REPRODUCTION命令|参数/文件不变；完整12层Na/Nf|covered_with_scope|
|A12|Encoder和全模型FLOPs公式与实际长度|§7/8 图4|[components](../scripts/analyze_costs.py#L24)|E6；见REPRODUCTION命令|Encoder及全模型，实际G/G16分开|covered_with_scope|
|A13|打分、选择等额外开销|§8/16|[select_tokens](../scripts/encoder_topk.py#L34)|E6；见REPRODUCTION命令|score/cast/排序/移动/同步单列范围|covered_with_scope|
|A14|pre-hook证明被丢弃token未进入当前FFN和后层|§12|[main](../scripts/profile_context.py#L21)|E3；见REPRODUCTION命令|Text5 prehook当前/下一层|covered_with_scope|
|A15|profiler矩阵shape与耗时；Multi每阶段覆盖|§12|[encoder_trace](../scripts/release_profile.py#L24)|E3/E7；见REPRODUCTION命令|Multi三阶段shape与CPU/CUDA时间|covered_with_scope|
|A16|从视觉到Projector到LLM的实际传递|§5 图1|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|Text5 13块→832视觉位置保持|covered_with_scope|
|B1|Linear特征映射与token数量|§5|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|Linear pointwise通常保持N；概念说明|covered_with_scope|
|B2|MLP映射与token数量|§5|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|MLP pointwise通常保持N；未额外实现|covered_with_scope|
|B3|Q-Former映射与token数量|§5|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|BLIP-2学习query汇聚；仅概念来源|covered_with_scope|
|B4|Perceiver Resampler映射与token数量|§5|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|Flamingo视觉固定学习latent重采样；仅说明|covered_with_scope|
|B5|结合代码解释指定模型真实模块|§5|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|原HF SimpleMLP单Linear；pixel shuffle改N|covered_with_scope|
|B6|如何处理剪枝后视觉特征|§5|[restore_grid](../scripts/encoder_topk.py#L62)|E2/E3；见REPRODUCTION命令|postnorm后零网格，原Connector不改|covered_with_scope|
|C1|自选三层同时跳FFN、无需训练|§5|[skip_forward](../scripts/ffn_skip.py#L16)|E4；见REPRODUCTION命令|三层8/16/24同时Skip|covered_with_scope|
|C2|支持Skip列表配置|§5|[__init__](../scripts/ffn_skip.py#L33)|E4；见REPRODUCTION命令|可配置列表/范围/重复检查|covered_with_scope|
|C3|残差、归一化处理正确|§5|[skip_forward](../scripts/ffn_skip.py#L16)|E2/E4；见REPRODUCTION命令|保Attention/inputnorm/cache；省norm2/MLP|covered_with_scope|
|C4|解释选层理由|§5|[skip_forward](../scripts/ffn_skip.py#L16)|E1；见REPRODUCTION命令|预先分散，未宣称最优|covered_with_scope|
|C5|实际存储参数与被跳FFN参数、统计方法|§5|[load_original_model](../scripts/run_original.py#L166)|E4；见REPRODUCTION命令|MLP7962624、norm2另1728；权重未删|covered_with_scope|
|C6|FLOPs公式与prefill/decode/T/G范围|§8|[components](../scripts/analyze_costs.py#L24)|E6；见REPRODUCTION命令|15925248(T+G−1)，prefill+G−1decode|covered_with_scope|
|C7|调用记录与profiler证明真正未执行|§11|[skip_trace](../scripts/benchmark_efficiency.py#L163)|E4/E7；见REPRODUCTION命令|零调用+投影marker缺席；v3 180→162|covered_with_scope|
|V1|基线+四Encoder+Skip，12组|§6|[patch_for](../scripts/experiment_runtime.py#L63)|E1；见REPRODUCTION命令|六配置两集3072；不叠加方法|covered_with_scope|
|V2|单点100%、Multi全100%，关闭额外打分|§11|[correctness](../scripts/release_checks.py#L50)|E2/E7；见REPRODUCTION命令|8路径32张量，score/scatter0；不代替50%|covered_with_scope|
|V3|空Skip与原模型输出一致|§11|[empty_stage](../scripts/verify_skip.py#L94)|E2/E7；见REPRODUCTION命令|空Skip独立cache prefill/decode与IDs|covered_with_scope|
|V4|说明比较对象、浮点误差|§11|[equal](../scripts/verify_encoder_topk.py#L23)|E2/E7；见REPRODUCTION命令|maxerr0，atol/rtol0；Encoder未独立decode logits|covered_with_scope|
|V5|小样本hook/profiler标配置/样本/模块|§12|[main](../scripts/profile_context.py#L21)|E3/E4；见REPRODUCTION命令|配置/ID/模块明确；诊断不代替稳定延迟|covered_with_scope|
|V6|效率同硬件/精度/batch/生成设置|§2/9/10|[generation_block](../scripts/benchmark_efficiency.py#L80)|E6；见REPRODUCTION命令|同硬件BF16/batch1/eager；不混版本|covered_with_scope|
|V7|计时范围/预热/同步/n/重复/G|§9/10|[timed](../scripts/benchmark_efficiency.py#L65)|E6；见REPRODUCTION命令|旧780/312；n/G/边界/同步/重复明确|covered_with_scope|
|V8|说明Attention获取及后端影响|§3/16|[prune_layer](../scripts/encoder_topk.py#L153)|E2；见REPRODUCTION命令|显式eager权重；Flash/SDPA未测|covered_with_scope|
|V9|FLOPs约定与范围、Encoder/prefill/decode分开|§7/8|[components](../scripts/analyze_costs.py#L24)|E6；见REPRODUCTION命令|主要矩阵Conv乘加2，辅助范围明示|covered_with_scope|
|V10|理论量与实测不一致的证据分析|§10/16|[stage_block](../scripts/benchmark_efficiency.py#L137)|E6；见REPRODUCTION命令|自然G/选择回填开销；独立median不相加减|covered_with_scope|
|V11|结果如实，不强求精度/速度提升|§6/16|[score_response](../scripts/check_scores.py#L59)|E1/E6；见REPRODUCTION命令|保留Multi/Skip负结果与无存储收益|covered_with_scope|
|S1|代码仓库或压缩包|§14/15|[main](../scripts/evaluate_release.py#L18)|E7/E8；见REPRODUCTION命令|公开main与固定Release已发布；当前状态见PUBLICATION；两者版本不同|published_with_scope|
|S2|完整PyTorch实现、六配置、依赖、模型版本|§2/3/5|[patch_for](../scripts/experiment_runtime.py#L63)|E1/E8；见REPRODUCTION命令|源码六配置/依赖锁/版本齐全|covered_with_scope|
|S3|数据准备命令与固定样本清单|§14|[prepare_manifests](../scripts/inspect_data.py#L71)|E1/E7；见REPRODUCTION命令|两数据路径职责明确；冷HF重建未测|covered_with_scope|
|S4|评估运行命令|§14|[main](../scripts/evaluate_release.py#L18)|E8；见REPRODUCTION命令|新24预测与聚合已测；limit256未跑|covered_with_scope|
|S5|正确性和性能分析运行命令|§14/15|[main](../scripts/release_checks.py#L143)|E7；见REPRODUCTION命令|正确性/性能/profiler/CPU命令；v3 first/stages未跑|covered_with_scope|
|S6|报告结果表、必答、实现、限制|§1至17|[page](../scripts/build_report.py#L22)|E1至E8；见REPRODUCTION命令|完整Markdown/PDF/Word同源；渲染见验收记录|covered_with_scope|
|S7|hook/调用/profiler执行证据|§11/12|[skip_trace](../scripts/benchmark_efficiency.py#L163)|E2/E3/E4/E7；见REPRODUCTION命令|精选proof可点；完整trace在科学包|covered_with_scope|
|S8|表中配置/层号/比例/K/最终比例/Skip层|§3 配置表|[config_path](../scripts/experiment_runtime.py#L20)|E1；见REPRODUCTION命令|层/阶段率/K/最终率/Skip表|covered_with_scope|
|S9|两数据集分数及相对基线分差|§6 图3|[read](../scripts/build_figures.py#L21)|E1；见REPRODUCTION命令|两指标与同版pp；解析失败保留分母|covered_with_scope|
|S10|总参数/跳过FFN参数/权重大小|§5|[load_original_model](../scripts/run_original.py#L166)|E4；见REPRODUCTION命令|总参数/停用参数/文件大小|covered_with_scope|
|S11|逐层Encoder/Projector入出/LLM视觉token|§4/5 图1/2|[connect](../scripts/encoder_topk.py#L180)|E3；见REPRODUCTION命令|逐层Na/Nf与Projector/LLM形状|covered_with_scope|
|S12|Encoder/全模型FLOPs、延迟、相对变化与实际G|§7至10|[components](../scripts/analyze_costs.py#L24)|E6；见REPRODUCTION命令|Encoder/全模型FLOPs、延迟/G/同版比|covered_with_scope|
|S13|随样本变化时给代表明细和子集汇总、说明统计|§13|[csvread](../scripts/build_figures.py#L24)|E1/E6；见REPRODUCTION命令|Text5案例、全256成本范围、小n性能统计|covered_with_scope|
|T1|说明外部代码来源|§17|[EncoderTopKPatch](../scripts/encoder_topk.py#L75)|E5；见REPRODUCTION命令|来源/license/固定commit与必要适配|covered_with_scope|
|T2|说明AI用途与自行核验关键内容|§17|[main](../scripts/evaluate_release.py#L18)|E8；见REPRODUCTION命令|AI实际用途与自动检查明确；不冒称人工|covered_with_scope|
|T3|可说明错误修正，无需全聊天|§17|[validate_protocol](../scripts/context_runtime.py#L25)|E5/E7；见REPRODUCTION命令|历史勘误/失败保留；不发布完整聊天|covered_with_scope|
|T4|理解并对实现/数据/结论负责|§17|[main](../scripts/evaluate_release.py#L18)|E8；见REPRODUCTION命令|个人理解/责任不能由AI文件证明；尚未验收|not_assessed|
|T5|30—45分钟技术交流：代码、结果、小范围现场修改|§17|[select_tokens](../scripts/encoder_topk.py#L34)|E8；见REPRODUCTION命令|30分钟slides/讲稿/代码/练习是准备材料；交流未发生|prepared_future_event_unverified|
|T6|指定现场环节不用AI，可查代码文档|§17|[select_tokens](../scripts/encoder_topk.py#L34)|E8；见REPRODUCTION命令|离线索引齐全；现场指定不用AI需用户未来执行|prepared_future_event_unverified|
