# 来源、必要改动与包依赖（postday3-v3）

|模块|固定上游/文件/许可|实际关系与本地必要改动|验证与边界|
|---|---|---|---|
|模型/Processor/native generate|HF SmolVLM-256M-Instruct@7e3e67edbbed1bf9888184d9df282b700a323964，模型卡Apache2.0|依赖调用；固定预处理/模板、CUDA/eager/BF16/seed和记录|原质量/规范性能；新包验收另列，不复制模型|
|Idefics3/Llama/Processor|Transformers4.57.6，上游753d61104116eefc8ffc977327b441ee0c8d599f；Apache2.0|安装源码未改；encoder_topk实例改Encoder循环/插入gather、postnorm后网格适配；ffn_skip实例省norm2+MLP|保存4份原生源SHA、100%/空Skip、数值、hook/profiler；旧源码原字节|
|PyTorch|2.7.1+cu128，torchvision0.22.1+cu128；BSD风格各自随源许可|直接调用tensor/native profiler/CUDA event/计时；不重写算子|原始profiler/性能记录|
|TextVQA|MMF@72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py；BSD|vendor原样复制，调用AnswerProcessor/Evaluator；未装完整框架|1536输出闭式聚合差异0；normalizer同源|
|ScienceQA|lupantech/ScienceQA@2cbf8318e07b9ece895bb2ae605e71e38d623264/tools/evaluate_acc.py；MIT|指标定义参考，实际子集分母256；extract_choice本地label-blind解析|解析失败0分保留分母；lecture/solution不参与|
|交叉评分参考|lmms-eval@3d12c1760482c9753b1c6416348228fef83bd0dd/lmms_eval/tasks/_task_utils/vqa_eval_metric.py；分范围MIT/Apache说明及共同来源|AST提取同源normalizer运行；独立闭式留一聚合；泛VQA@a013f0043c1e2cdc995922dfe257f7149aa9af06只阅读|scoring/sources保存原字节/license，不说异源完整评分器|
|Token参考|题目论文2504.01690v2；AudioMAE相关仓库|仅机制参考，未复制/执行音频框架；题目要求原序/多块/mask由本地最小适配|未引入训练/timm环境|
|Projector概念|LAVIS@05cb670f47296522e24e7727e366b4a997d28582（文件头BSD3，全文404）；DeepMind@f5de0ede8430809180254ee957abf36ed62579ef（Apache2）；LLaVA@c121f0432da27facab705978f83c4ada465e46fd（Apache2）|只阅读/引用链接，不是模型依赖；包内不分发这三份未执行参考py，历史源/404保留原项目，provenance及已有license记录保留包内|无需额外框架或许可下载阻塞；没有声称三份全文全取得|
|成本/性能串接|项目已有analyze_costs/benchmark_efficiency及原HF/helpers|必要计时/离线统计；新包CPU直接复用现入口；GPU是否需要适配以实际拒绝记录为依据|不重做3072质量/780性能；不建立新框架|

完整固定URL/SHA/license文件见evidence/day1_morning/sources、scripts/vendor/README.md，以及evidence/day2_evening/cost_perf_v1/{scoring/sources,references}/provenance.json。参考provenance里的3个概念py本包故意不附，其取舍明确写入打包排除项；原项目历史文件不删除。

包必须保留：evidence/day1_morning/revisions.json与sources/model_files_metadata.json（原loader直接读取）；旧v1协议/父contract/父配置；v2配置与各SHA；两manifest/meta及512实际图像；原正式raw/IDs/reference和模型runtime config；旧B2/B3/B4/B5/B6证明/失败记录与当前成本/性能/评分记录。删除这些父依赖可能使加载或核验失败。旧硬环境指纹是原路径历史链，不通过伪造fingerprint或关闭assert迁移。

AI承担主要适配/编排/统计/报告/自动核查。用户提供任务/范围决策；不声称未参与的人工操作已完成。个人学习目标暂缓。模型/Processor/权重由固定HF版本取得，本次包验收允许显式共享原解释器和HF热缓存；不分发.venv或权重，不声称跨机冷安装。

## 实际新增重放入口

v2新增推理代码scripts/replay_release.py（149行），用于候选原入口实际exit1后的路径适配；其必要性/复用方案/最小改动/验收见REPLAY_ADAPTATION。无模型、Processor、generate、评分公式或六套框架复制。核心旧源码逐文件未变，patch_for保留六配置分发，manifests_for验证固定抽样，run_sample执行原生处理/生成/评分。包内现环境独立绑定、旧ID只历史参考；完整原source/config/manifest SHA与实际原生源版本仍校验。RC1解包12条通过，limit256能力未全跑。新本地代码不额外宣称整个仓库统一许可，上游头部/许可按文件保留。

## v3新增执行编排

release_checks.py复用replay真实binding、旧sample_run/Observe/equal/empty_stage与benchmark测量块。release_profile.py由旧profile_context.main逐配置插桩块机械提取，只参数化输入/配置/输出并改trace绝对路径。新增设置和固定诊断清单不改变旧协议。两文件均为本地适配，不将它们称为上游原生代码；没有复制模型、Processor、generate、评分或六套评估循环。必要性与验收见[适配记录](RELEASE_V3_ADAPTATION.zh.md)，实测范围见[能力矩阵](COMMAND_CAPABILITY_V3.zh.md)。
