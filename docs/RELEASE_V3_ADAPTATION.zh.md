# v3 必要适配（实施前记录）

具体缺口：v2 replay 已能迁移推理；旧 correctness/benchmark/profile main 绑定历史根，无法作为解包根运行入口。

现成复用：replay_release.binding_for 核科学 SHA、真实依赖、原生 Transformers 源及当前环境；context loader/builder；verify_context.sample_run、Observe/equal/toy_checks/verify_path；verify_skip.empty_stage；benchmark_efficiency.prepare/generation_block/stage_block/skip_trace；原 profile_context 的逐配置插桩及算子提取块。

最小修改：新增 release_checks.py 只编排模式/路径/绑定/清理；release_profile.py 从原 profile_context.main 机械提取逐配置块，改输出路径并参数化，不复制模型或评测循环。新增独立性能设置与原诊断样本清单。旧脚本、协议、六配置、图片、3072质量、780性能和proof原字节保留。

验收：先打RC并独立解包；检查512图片；Text5/Science2的四Encoder全100%及空Skip；两诊断输入六配置natural/fixed16共120正式48预热；首尾baseline锚点另列；Text5 Multi三阶段和baseline/Skip短trace。所有新输出在包外，新绑定含入口/helper源码SHA及输入/参考真实路径。共享解释器/GPU/HF缓存明示。不会伪造历史environment_id或改变断言。新记录只证明此范围，不是跨机冷装或全量重跑。
