# 最终复现说明

以包根[README](../README.md)的命令为唯一有效新重放入口；环境/源码/数据说明及旧入口边界见该文。

# 历史原路径复现记录（保留，不是包内迁移命令）

项目绝对根 `/home/zx/Documents/Shen-Wu-Challenge`。已有12组3072质量记录直接使用，旧质量152冻结与B6最终验收引用不能更改。新成本/性能全在 `/home/zx/Documents/Shen-Wu-Challenge/evidence/day2_evening/cost_perf_v1`。本说明区分查看、同机重放和未验证跨机。

## 1. 实际版本与前提

`.venv/bin/python` → /usr/bin/python3.10；3.10.12 2026-08-31构建；torch2.7.1+cu128/transformers4.57.6，RTX5090 Laptop24463MiB/driver580.178.04；BF16 eager batch1，pixel_values保持正式FP32。环境ID98f15960cb99ee3d7bae5d7713b1bc7c44471dde0831dab1aa2d7a1729b91531。其余锁定版本看requirements-morning.lock.txt与下午b5/packages.json；本轮未重装依赖。

模型HF revision7e3e67edbbed1bf9888184d9df282b700a323964，权重SHA74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e、513028808 bytes。模型/Processor/config/tokenizer通过原HF from_pretrained同revision加载；现有HF_HOME=data/hf_cache，offline运行，无网络模型获取。

数据：TextVQA lmms-lab-encoder/textvqa@9c0699cd19768ac5ab97568f6b3cbac4c0062884、ScienceQA-IMG lmms-lab/ScienceQA-IMG@b9984af01798586705cee1fbdfa80a6dcbef04f1，default/validation。data/manifests下两256清单SHA分别6bf38f9c78d83bbdb86f0ba18c3a1ea45cd9977c7ad3ba576cf9a679080eea53、7c684cc91e680f9098e1f8fbfd25081ca858e02172b743abdc5c2c075623fa3a；data/baseline_inputs下512图片。既有inspect_data.py --prepare-manifests PROTOCOL可按固定seed/revision准备，本轮不执行重抽样/下载；冷缓存需HF Datasets指定revision及HF hub同revision获取，再另做环境绑定/小样本验收，未验证一键跨机安装。

## 2. 查看已有结果（不会generate）

直接读 reports/REPORT.zh.md、reports/REQUIREMENTS_MATRIX.zh.md；旧质量b6/RESULTS.zh.md；本轮costs/summary.json与performance_summary.json；各phase/config的JSONL是原始重复，warmup必须过滤。

不要把旧audit_results.py、evaluate_context.py --summarize-only当完全只读：前者覆盖审计/CSV/RESULTS并使旧final_verification引用过期，后者重写汇总。新分析都写新目录。

## 3. 实际CLI与主要命令

以下参数已经 --help 检查，主测量已实际exit0；从项目根执行。

```bash
cd /home/zx/Documents/Shen-Wu-Challenge
.venv/bin/python -B scripts/benchmark_efficiency.py --help
.venv/bin/python -B scripts/analyze_costs.py --help
HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 .venv/bin/python -B -u evidence/day2_evening/cost_perf_v1/run_E2.py
HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 .venv/bin/python -B -u evidence/day2_evening/cost_perf_v1/run_A1.py
.venv/bin/python -B evidence/day2_evening/cost_perf_v1/summarize_perf.py
```

控制器按configuration baseline/skip/early/middle/late/multi串行；phase含natural、first、fixed16、probe、stages、skip_trace、anchor_start/end、smoke。已有complete只在绑定完全相同才跳过；存在partial不会覆盖，保留原尝试后用新 --output-root。成本输出目录非空会拒绝，防止重写旧结果。不要并发两个GPU控制器；新入口共享旧gpu_lock并检查外部compute PID，不能阻止未配合的外部程序，仍需检查。

## 4. 按本说明实际执行的独立交付冒烟

下面的目录是新建输出，不混入旧质量；选择固定诊断TextVQA原索引1848。依次baseline、Early、Skip，每个新进程原生generate一次，与各自正式完整new_token_ids精确一致。成本从全部保存记录重算到新输出，不再次推理3072条。

```bash
HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 .venv/bin/python -B -u scripts/benchmark_efficiency.py --configuration baseline --phase smoke --output-root evidence/day2_evening/cost_perf_v1/delivery_smoke
HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 .venv/bin/python -B -u scripts/benchmark_efficiency.py --configuration early --phase smoke --output-root evidence/day2_evening/cost_perf_v1/delivery_smoke
HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 .venv/bin/python -B -u scripts/benchmark_efficiency.py --configuration skip --phase smoke --output-root evidence/day2_evening/cost_perf_v1/delivery_smoke
.venv/bin/python -B scripts/analyze_costs.py --output-dir evidence/day2_evening/cost_perf_v1/delivery_smoke/costs
```

A4验收以delivery_smoke/verification.json记录的实际command/exit/输出SHA与比较为准，不能只看本段文字。再次执行成本需要另一个不存在的输出目录；不删除本次目录重跑。

## 5. 精确质量链与新性能链

质量入口仍为scripts/evaluate_context.py，role evaluation、6configuration、每run两manifest完整256。固定v2/context配置、B2/B3/B4/B5门禁使旧入口不能任意迁移。当前完整无需恢复；真实异常时同绑定--resume，持久化error经诊断才--retry-errors；改源/环境必须新版本/新run重验，不能改旧hash/关闭assert/用python -O。

新主要两个脚本复用context_runtime.load_context_model/build_question、experiment_runtime.configs/patch_for/gpu_lock、HF generate和verify_skip.prefill/decode。phase/新代码/新config/sample SHA、旧scientific binding均落盘，源副本每phase/config保存。外部hooks仅probe/trace启用，全named_modules恢复核对，主计时无外部hooks。

只读读取原记录的成本表包含实际G与G*=16、全模型主要矩阵/Conv，乘加2；辅助算术/移动范围单列。性能自然/首token8题、固定/阶段2题，各2预热5重复，不含load/Processor/H2D/I/O；single-decode wrapper包含mask/position构造，prefill在外。峰值allocated与reserved各自解释；stage captured buffers在GPU，不用其显存代替整请求。

## 6. 迁移与打包边界

路径BASE、旧E0 environment fingerprint、原始质量参考、父配置SHA和缓存目录均有本机/本批依赖。--output-root只参数化新增输出，不意味着全部工程已跨机。转移项目路径、解释器构建或torch/源文件变化需要新环境登记与小桥接，不能篡改旧hash通过。

获取方式：使用HF仓库精确revision获取完整模型/Processor文件与指定Datasets版本，或保留原固定manifest/图片且验证其hash；不盲目复制venv，不把513MB权重塞入每份报告。当前未做跨机冷缓存安装；教授若要离线交付，另确认权重/图片许可与体积。打包清单见reports/DELIVERY_MANIFEST.md。所有负结果/失败/原型记录保留。AI完成主要工程与草稿，本人独立能力尚未验收。

为显式记录离线成本命令的开始/完成/退出状态，又在delivery_smoke/costs_recorded目录执行一次CPU重放；两份CSV与主成本表字节一致。额外GPU冒烟没有重复，仍为三配置各一题；准确命令与时间见delivery_smoke/verification.json。


## 最新勘误

教授原文已保存docs/ASSIGNMENT.zh.md；历史原路径命令是真实成功范围，不能直接从迁移包运行旧指纹入口。新包GPU用replay_release.py，科学配置/旧源仍验证，当前环境独立绑定；CPU用原analyze_costs.py显式probe-root与包外新输出。实际包内成功范围以validation记录为准。个人学习目标暂缓，不作为交付阻塞。

## 本次已执行的包内范围

原成本入口从RC1解包路径成功重算两CSV，SHA与历史相同；新replay六配置各两首题共12条通过。实际命令见../validation/RC1_commands.json、回执见../validation/RC1_acceptance.json。代码/配置/图片/历史参考均来自解包根；共享外部解释器/依赖/HF热缓存明确记录。失败legacy_attempt日志和记录保留，新源没有改旧fingerprint/门禁。最终包解包检验由包外FINAL_RECEIPT.json给出；不称跨机冷安装。
