# postday3-v3 简短复现命令页

从实际解包根执行，固定依赖见锁文件。当前验收共享原机Python3.10.12、torch2.7.1+cu128、Transformers4.57.6、RTX5090 Laptop GPU及HF热缓存，不含跨机冷安装。代码/配置/512图片/正式参考/proof必须来自解包根，HF缓存只提供固定revision模型/Processor。

```bash
cd /path/to/unpacked/Shen-Wu-Challenge
P=/home/zx/Documents/Shen-Wu-Challenge/.venv/bin/python
export HF_HOME=/home/zx/Documents/Shen-Wu-Challenge/data/hf_cache
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
OUT=/tmp/shenwu-v3-new-run
"$P" -B scripts/release_checks.py --help
```

每个输出目录必须为空且位于包外；重复运行请更换OUT，失败不覆盖。不可用python -O。以下参数与真实help、成功命令一致；精确argv/cwd/起止/退出在validation/postday3_v3/commands.json，日志在其logs目录。

|模式|命令|读取|写出／本次实际范围|
|---|---|---|---|
|数据就绪|`"$P" -B scripts/release_checks.py --mode data --output-dir "$OUT/data"`|两固定manifest/meta、512PNG、冻结协议/来源|data_ready与binding；全部512 SHA，未重采样/下载全数据|
|评估重放|见下方六配置循环|包内manifest/图片、对应正式参考、六配置|每集首题；六配置×2=12条全字段对齐；limit1..256可表达，256未重跑|
|正确性|`"$P" -B scripts/release_checks.py --mode correctness --output-dir "$OUT/correctness"`|Text5/Science2、原HF与原patch|四Encoder100%×2，Encoder/postnorm/Connector/prefill logits和完整IDs；空Skip×2 prefill/独立cache一步decode/完整IDs；清理检查|
|性能|见下方phase/config循环|Text1848/Science211、独立v3设置、旧正式参考|每组合2预热5正式；natural/fixed16共120正式48预热；首尾锚点另20正式8预热|
|profiler|`"$P" -B scripts/release_checks.py --mode profiler --output-dir "$OUT/profiler"`|固定Text5、旧插桩逻辑|Multi三阶段及后层矩阵shape/time、baseline/Skip各prefill+1decode、调用与Chrome trace；独立于规范计时|
|CPU成本|`"$P" -B scripts/analyze_costs.py --probe-root "$PWD/evidence/day2_evening/cost_perf_v1" --output-dir "$OUT/cost"`|包内旧probe/配置/质量/Encoder长度|两CSV各3072行，已重算且字节匹配；无新模型推理|

```bash
for c in baseline skip early middle late multi; do
  "$P" -B -u scripts/replay_release.py --configuration "$c" --limit 1 --output-dir "$OUT/replay/$c"
done
"$P" -B -u scripts/release_checks.py --mode performance --configuration baseline --phase natural --output-dir "$OUT/anchor_start"
for phase in natural fixed16; do
  for c in baseline skip early middle late multi; do
    "$P" -B -u scripts/release_checks.py --mode performance --configuration "$c" --phase "$phase" --output-dir "$OUT/${phase}_${c}"
  done
done
"$P" -B -u scripts/release_checks.py --mode performance --configuration baseline --phase natural --output-dir "$OUT/anchor_end"
```

performance的first/stages参数可用但本轮未运行；stages复用原probe+模块回放和fresh-prefill单步decode块。100%不是50%路径证明：旧未变源码与b2/b4完整证据继续引用，v3 Multi trace与六配置12条重放为定向补查。正确性容差沿用旧函数atol=rtol=0，真实最大误差0。

性能计时：已在GPU的输入→原native generate返回并同步；不含加载/Processor/H2D/评分/I/O/外部hooks/profiler，生产补丁开销保留。natural完整IDs逐次对齐正式记录，fixed16逐次检查G=16；wall为主，CUDA Event为辅助。每重复JSONL保留wall/G/峰值/reserved/IDs，两次硬件快照包围每输入7重复，硬件查询在计时外，不宣称逐kernel硬件追踪。summary按每题5次线性25/75分位median/IQR计算，仅同新版本同题baseline配对，不拼接旧毫秒。

新证据见validation/postday3_v3；旧780主性能、3072质量与proof不改。limit256、first/stages本轮未运行，完整旧入口仍带历史门禁；历史命令长页见[历史附录](REPRODUCTION.v2-history.zh.md)，不作为当前包根命令。性能小样本、同源normalizer、官方73.6未复现、负质量结果与无显著省显存/存储限制仍有效。
