# 接收者复现说明

本页适用于“轻量代码仓库 + postday3-v3完整科学包”。Git中的科学脚本/配置与包内对应文件原字节一致；新增 `scripts/evaluate_release.py` 只负责新预测评估，实际导入全部绑定到显式 `--artifact-root`，不会混读仓库之外的作者工程。

## 1 支持范围与独立环境

已测：Linux x86_64、Python 3.10.12、torch 2.7.1+cu128、torchvision 0.22.1+cu128、Transformers 4.57.6、RTX 5090 Laptop 24463 MiB、驱动580.178.04、CUDA可用且支持BF16。batch=1、eager。没有CPU推理回退；Windows/macOS和其他GPU未验证。Python构建/机器信息作为当前环境登记，科学源码/原生Transformers四模块SHA和依赖版本保持严格检查。不要使用 `python -O`。

本轮使用已有实验解释器与HF热缓存，没有重新安装或冷下载。以下独立安装命令为接收者路径，未在本轮实装；不保证未来软件源永远保留这些包。依赖锁为 [requirements-morning.lock.txt](../requirements-morning.lock.txt)，它是实测快照，不是所有平台的通用兼容性声明。

```bash
# 从轻量仓库根执行。准备 Python 3.10 后创建独立环境。
python3.10 -m venv .venv-receiver
P="$PWD/.venv-receiver/bin/python"
"$P" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
"$P" -m pip install -r requirements-morning.lock.txt
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
```

## 2 完整科学包与模型缓存

完整包文件名 `Shen-Wu-Challenge-postday3-v3.tar.gz`，455733348 bytes，SHA256 `d95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d`。本地候选可从同交付目录的 publication 清单定位。线上发布状态见 [PUBLICATION](PUBLICATION.zh.md)，未发布时没有可用的公开下载URL。

把获得的原包放在仓库根，先校验再解包，不修改封存根：

```bash
echo 'd95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d  Shen-Wu-Challenge-postday3-v3.tar.gz' | sha256sum -c -
mkdir -p artifacts
tar -xf Shen-Wu-Challenge-postday3-v3.tar.gz -C artifacts
ARTIFACT_ROOT="$PWD/artifacts/Shen-Wu-Challenge"
OUT="$PWD/runs/receiver-$(date +%Y%m%d-%H%M%S)"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE
```

`HF_HOME` 可换成已有缓存目录。空缓存必须先联网获取固定revision，不能先开offline再声称准备完成。`snapshot_download` 为已安装 huggingface_hub 0.36.2 的实际API，revision/local_files_only参数已核实；以下下载本轮未重复执行。

```bash
"$P" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct',
    revision='7e3e67edbbed1bf9888184d9df282b700a323964'))
PY
```

下载整个固定模型快照包含权重、模型配置、Processor、tokenizer和chat模板。实际loader另外核对原权重LFS SHA、Processor设置和源码SHA。准备完成后可用 `local_files_only=True` 的同一API确认本地快照；这只查缓存，不等于GPU推理验收。之后才设置：

```bash
export HF_HUB_OFFLINE=1
```

参数依据：[HF snapshot 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。联网下载/独立安装未在本轮测试，不能用同机热缓存24条担保冷启动。

## 3 两种数据路径

推荐直接使用原包 `data/manifests/*256.jsonl`、meta和 `data/baseline_inputs/{textvqa,scienceqa}/*.png`。两份清单各256，512张PNG保存RGB原尺寸；模型输入缩放由固定Processor完成。种子20261029、validation、固定revision与ID已写入清单。

确需从HF重建时，原入口是 `scripts/inspect_data.py --prepare-manifests configs/eval_protocol_context_v2.json`。它以**脚本所在根**写 `data/manifests`、`data/baseline_inputs`；没有output-root参数。已有清单时只校验，不重建；缺清单时读取 `data/diagnostic/*_sample.json` 核对HF validation shard顺序。不能在封存根直接运行重建，也不要漏掉这些诊断父依赖。

安全的历史重建用法是在原包的独立副本中执行，并只在该副本删除两manifest/meta与baseline_inputs后重建（不动原包、U及其记录）。本轮未做这条HF重建；PNG编码/依赖差异可能改变SHA，不能以修改旧meta/hash强行通过。先比较新旧ID、字段、像素与字节并另存结果，再决定是否可作为相同输入使用。

环境和模型完成后，运行随包就绪校验。它读取冻结依赖、协议和GPU信息，并核对两清单/512PNG；它不是纯CPU下载器：

```bash
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" --mode data --output-dir "$OUT/data"
```

数据模式本轮未重复执行；此前v3回执的512SHA和20261031只读审查均保留。新24条对实际使用的4张图重新核SHA。

## 4 新预测评估

这是本轮新增的薄入口，复用 `binding_for`、`load_context_model`、`build_question`、`patch_for`、`run_sample`、`summarize` 和原评分器。所有科学模块只从ARTIFACT_ROOT/scripts导入。历史绑定用于校验科学源/协议/清单与依赖，当前Python/GPU环境另存，历史环境ID只作引用。

```bash
# 实际验证命令模式：六配置各两集前2题，共24条。
for c in baseline early middle late multi skip; do
  "$P" -B -u scripts/evaluate_release.py --artifact-root "$ARTIFACT_ROOT" \
    --configuration "$c" --limit 2 --history-compare --output-dir "$OUT/new/$c"
done
# 接收者主动做完整评估时把 --limit 2 改为 --limit 256。
# 新入口完整256未在本轮执行，不能据此称本轮做了3072条。
```

每配置模型加载一次，两集串行；GPU互斥并拒绝外部compute任务。输出目录必须全新、位于artifact-root之外；失败目录不能覆盖，不提供自动续跑或自动删错重试。逐题保存原始generate输出、new IDs、prompt/input IDs/shape、score/parse、T/G、配置、输入/协议SHA和binding。异常保留记录并使运行未完成。解析失败记0，保留请求分母。摘要保留 `mean_score=null`（partial时）、`partial_mean_score`、`requested_mean_score`、requested_target/inferred_count/errors及partial标志；不完整请求不出完成分数。

`--history-compare` 单独输出差异诊断，不强制新生成与旧new IDs相同；可省略。当前严格固定Transformers源码与依赖，尚未支持任意新版本软件。科学包仍是必要父依赖，不是克隆轻量Git即可独立推理。

实际验收：[NEW24_ACCEPTANCE](../verification/NEW24_ACCEPTANCE.json)、[精确命令](../verification/commands.json)。真实24条均完整、无异常/解析失败、所有比较字段与历史相等。这是同机热缓存小样本验证，不增加主质量表。

## 5 精确重放与正确性

```bash
# 精确重放：要求每个记录字段与历史相同。与新预测评估职责不同。
"$P" -B "$ARTIFACT_ROOT/scripts/replay_release.py" --configuration early \
  --limit 1 --output-dir "$OUT/replay-early"
# 正确性：100%/空Skip、独立cache decode、清理和受控异常。
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" --mode correctness \
  --output-dir "$OUT/correctness"
```

以上以旧v3保存的实测回执为证据，本轮未重复。100%Encoder只单独比较prefill logits，完整生成IDs也相等；空Skip另含独立cache一步decode。它们不代替真实50%和Multi的hook/profiler证明。

## 6 性能与profiler和CPU成本

```bash
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" --mode performance \
  --configuration baseline --phase fixed16 --output-dir "$OUT/perf-baseline-fixed16"
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" --mode profiler \
  --output-dir "$OUT/profiler"
"$P" -B "$ARTIFACT_ROOT/scripts/analyze_costs.py" \
  --probe-root "$ARTIFACT_ROOT/evidence/day2_evening/cost_perf_v1" --output-dir "$OUT/costs"
```

性能每次两集各1诊断题，2预热5正式；配对比较应在同一运行版本增加同模式的六配置、首尾baseline锚点。GPU上输入到native generate返回并同步，不含加载/Processor/H2D/评分/I/O。生产选择/回填/同步保留。`natural` 允许G变化，`fixed16` 控制G=16，后者文本不入质量表。`first/stages` 是可调用但v3未实跑的范围；旧cost-perf-v1有已保存结果。CPU成本只读取旧原子记录估算，不做新模型推理，不把重算称硬件FLOPs实测。

本轮不重跑旧780或v3 120性能，正文图使用旧780主基准原始记录重汇总，v3单独表述。完整版本和证据入口见 [EVIDENCE_INDEX](EVIDENCE_INDEX.zh.md)。
