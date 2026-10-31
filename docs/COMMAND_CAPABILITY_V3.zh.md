# 命令能力矩阵 · postday3-v3

当前实测根：delivery/postday3_v3/unpacked_rc1/Shen-Wu-Challenge（精确绝对路径与argv在validation/postday3_v3/commands.json）。共享外部解释器/GPU/HF缓存；本次所有项目模块、配置、图像和正式参考均从该独立解包根读取。完整命令见[复现页](REPRODUCTION.final.zh.md)。

|教授§6要求|历史事实|当前包根入口／读写|本次实际验证|剩余边界|
|---|---|---|---|---|
|数据准备|固定两256 manifest/meta与512PNG|release_checks --mode data；读包内清单/图片，写包外data_ready/binding|512图像SHA、清单抽样/版本/meta全部通过|不重抽样/下载全数据；不含跨机冷缓存准备|
|评估|旧六配置12组3072质量保留|replay_release --configuration NAME --limit N --output-dir OUT；包内配置/图片/正式参考→新输出|新RC六配置×Text5/Science2=12条，prompt/IDs/raw/score/parse/长度全对齐|N=1..256可表达，仅N=1实跑；无新全量3072|
|正确性|旧b2含100%、50%、mask/残差、多阶段/Skip|release_checks --mode correctness；真实HF+原数值helper→包外proof|8个100%路径×4张量=32比较maxerr0；额外score/scatter0；2空Skip的prefill/独立cache decode、完整IDs；全模块清理、受控异常退出|100%未独立比较decode logits；真实50%依旧未变b2及当前12重放/Multi trace，不将旁路当选择证明；任意OOM/kill未测|
|性能分析|cost-perf-v1的780正式/312预热不变|release_checks --mode performance --phase natural/fixed16 --configuration NAME；原prepare/generation_block/timed→包外JSONL|Text1848/Science211×六配置×两模式×5=120正式、48预热；首尾baseline另20/8；natural IDs对齐、fixed16全G16；wall/event/显存/硬件快照齐全|仅两诊断题功能复现；first/stages可调用旧测量块，但本轮未运行；不和旧表拼毫秒/加速比|
|矩阵/profiler|旧b4与skip_trace保留|release_checks --mode profiler；原插桩块/skip_trace→包外trace/proof|Text5 Multi L3/6/9为512/256/128并覆盖后层；baseline/Skip prefill+1decode，MLP矩阵180/162，目标Skip为0|诊断profiler与规范计时分开；CPU/CUDA嵌套时间不相加；未重建旧所有配置trace|
|CPU理论成本|两CSV主表共6144行|原analyze_costs --probe-root 包内路径 --output-dir 包外|两3072行CSV重算SHA字节一致|读取旧结果估算，无新质量/硬件FLOPs测量|

所有27条候选命令exit0；模型/数据/性能验收失败为0；封装前排除规则误判1次，已保留日志并收紧检查，未重跑GPU。v2 legacy exit1与旧失败日志完整保留，不改旧passed或门禁。本轮R0—R4，不执行O1、训练/调层/比例扫描、个人问答与能力验收。最终包完整性由包外FINAL_RECEIPT.json确认；本页不是预先通过回执。
