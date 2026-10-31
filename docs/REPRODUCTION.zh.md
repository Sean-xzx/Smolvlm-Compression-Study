# 接收者复现说明

**顺序：clone阅读 → 下载指定科学附件 → 核SHA → 解包 → 准备固定依赖/模型 → 指定artifact-root运行。单独clone不含完整推理输入。** 本页是当前接收者入口，旧 `REPRODUCTION.final` / `v2-history` 是历史说明，版本关系见[PUBLICATION](PUBLICATION.zh.md)。

本次仓库改造仅核对命令与入口，**没有执行下面的安装、下载、GPU推理、性能或数据重建命令**。文中“已测”来自既有回执，不能把参数可传当作已经跑过。

## 1 Clone阅读与只读检查

```bash
git clone --branch main https://github.com/Sean-xzx/Smolvlm-Compression-Study.git
cd Smolvlm-Compression-Study
python3 -B scripts/check_repository.py
```

读[报告](../report/REPORT.zh.md)、[题目检查表](../README.md)、[证据索引](EVIDENCE_INDEX.zh.md)无需GPU。上述检查只用Python标准库、只读文件/本地链接/配置，不联网、不下载、不加载模型、不写验收。缺少完整包时会明确显示推理输入未检查；结构通过不等于科学正确性或环境就绪。

`scripts/validate_delivery.py` 是历史本机交付工装，依赖仓库外new24及旧发布阻塞快照，**不是接收者自检**；默认拒绝执行，不再默认写历史 `FINAL_ACCEPTANCE.json`。其历史逻辑/限定输出规则见 `--help` 和[发布说明](PUBLICATION.zh.md#本地工装与接收者检查)。

## 2 下载科学附件、核SHA并解包

从固定[正式Release](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/tag/autonomous-20261031-release)取得 **Shen-Wu-Challenge-postday3-v3.tar.gz**；455733348 bytes。Release的代码ZIP/自动源码包不能代替该科学包。已有已核验原包时直接复用，不必重复下载。

```bash
mkdir -p downloads
ARCHIVE="$PWD/downloads/Shen-Wu-Challenge-postday3-v3.tar.gz"
# 仅缺少该文件时执行下载；不要覆盖已有附件。
if [ ! -e "$ARCHIVE" ]; then
  curl --fail --location \
    'https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/Shen-Wu-Challenge-postday3-v3.tar.gz' \
    --output "$ARCHIVE"
fi
printf '%s  %s\n' \
  'd95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d' \
  "$ARCHIVE" | sha256sum -c -
# 只有校验成功才继续。目标必须全新；已有已核验解包根则跳过解包。
mkdir -p artifacts
test ! -e artifacts/Shen-Wu-Challenge && tar -xf "$ARCHIVE" -C artifacts
ARTIFACT_ROOT="$PWD/artifacts/Shen-Wu-Challenge"
OUT="$PWD/runs/receiver-$(date +%Y%m%d-%H%M%S)"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
python3 -B scripts/check_repository.py --artifact-root "$ARTIFACT_ROOT"
```

最后一条只读比对19项已保存的科学文件绑定，不核验全部图片、不测试GPU/模型或实验正确性。也可显式传 `--archive "$ARCHIVE"` 完整读取原包SHA，不会解包/执行；它是可选操作，不默认重复读大包。

完整包提供原科学helper/六配置/父协议、两manifest/meta、512张PNG、3072逐题结果、proof/trace及旧验证记录。当前Git中的科学副本保持相同字节，新增[薄入口](../scripts/evaluate_release.py#L18)实际从显式artifact-root导入科学模块，不混读原作者工程。只读检查器不替代原loader的严格绑定门禁。原 `release_checks.py` / `replay_release.py` 导入时就读取包内revisions父文件，因此即使只看 `--help` 也应调用artifact-root下的版本；直接运行轻量Git中的这两个副本会缺父依赖。

## 3 固定依赖与模型

既有实测平台：Linux x86_64、Python3.10.12、torch2.7.1+cu128、torchvision0.22.1+cu128、Transformers4.57.6、RTX5090 Laptop 24463MiB、驱动580.178.04，CUDA/BF16、batch1/eager。没有CPU推理回退；Windows/macOS/其他GPU未验证。依赖锁是实测快照，不是通用兼容性承诺；不要使用 `python -O` 绕assert。

以下独立安装步骤**未冷安装实跑**；既有新24使用原解释器及HF热缓存。

```bash
python3.10 -m venv .venv-receiver
P="$PWD/.venv-receiver/bin/python"
"$P" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$P" -m pip install -r requirements-morning.lock.txt
export HF_HOME="$PWD/.hf-cache"  # 可改为已有缓存目录
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE
"$P" -B - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct',
    revision='7e3e67edbbed1bf9888184d9df282b700a323964'))
PY
# 确认固定快照已完整取得后才开启离线。
export HF_HUB_OFFLINE=1
```

取得整个固定模型快照，包括权重、config、Processor、tokenizer和chat模板；loader另核对原权重LFS SHA、Processor设置、依赖及原生Transformers四模块源码SHA。`local_files_only=True` 的同一API可查本地缓存，但不等于推理验收；[HF下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)是参数来源。当前源码不支持任意升级后的软件环境。

## 4 数据准备与校验

推荐使用科学包现成的 `data/manifests/*256.jsonl`、meta与 `data/baseline_inputs/{textvqa,scienceqa}/*.png`。两集各256、共512张RGB原尺寸PNG；固定validation、revision、seed20261029和ID，实际输入缩放由固定Processor完成。

依赖/模型准备后，运行原包就绪校验：

```bash
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" \
  --mode data --output-dir "$OUT/data"
```

输入：科学包协议、manifest/meta、PNG与严格环境绑定；输出：新目录中的binding、`data_ready.json`、完成/失败记录。该命令包含CUDA/依赖就绪检查，**不是纯CPU下载器**。旧v3已有512图片SHA回执；本次未重复。新增24仅重新核验其实际用到的4张图。

如需从HF重建输入，只在原包的**独立可写副本**中进行，保留诊断父依赖；已有清单时原脚本只校验，不自动强制重建。需要重建时先在副本中移走旧manifest/meta和baseline_inputs，不能动封存根、原包或旧hash。

```bash
# DATA_COPY 必须指向另行准备的独立副本，不得等于 ARTIFACT_ROOT。
"$P" -B "$DATA_COPY/scripts/inspect_data.py" \
  --prepare-manifests "$DATA_COPY/configs/eval_protocol_context_v2.json"
```

原脚本按**脚本所在根**写清单/图片，没有output-root参数。HF重建本次未实跑；缺清单时还读取 `data/diagnostic/*_sample.json` 核对shard顺序。PNG编码/依赖差异可能改变SHA，应另存新结果比较ID/字段/像素/字节，不能更新旧绑定强行通过。

## 5 新评估

```bash
# 既有实测模式：六配置×两集前2题，共24条；本次不重跑。
for c in baseline early middle late multi skip; do
  "$P" -B -u scripts/evaluate_release.py --artifact-root "$ARTIFACT_ROOT" \
    --configuration "$c" --limit 2 --history-compare --output-dir "$OUT/new/$c"
done
# 完整评估可传 --limit 256，但新入口完整256尚未实跑。
```

输入：artifact-root中的固定代码/配置/数据/原绑定，及固定模型缓存；每配置加载模型一次、两集串行，GPU互斥并拒绝外部compute。输出：逐题raw/IDs/score/parse/T/G及输入/配置SHA、当前环境binding、各集摘要和complete/failure。输出必须是科学根外全新空目录，失败尝试保留，不自动覆盖或续跑。

复用 `binding_for/load_context_model/build_question/patch_for/run_sample/summarize`；`--history-compare` 仅保存诊断，可省略，不要求新预测与历史相等。解析失败记0保留分母；异常使运行未完成，不完整请求不出完整分数，partial摘要与旧主表隔离。

既有[新24验收](../verification/NEW24_ACCEPTANCE.json)及[原机器精确命令](../verification/commands.json)记录24条无异常/解析失败、历史字段相等。原argv/cwd/日志绝对路径保留为执行事实，接收者使用本页相对命令。完整新输出在[已发布new24附件](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/new24-verification-20261031.tar.gz)，不是新增3072主实验。

## 6 精确重放与正确性

```bash
# 严格重放：要求记录字段与历史相同，区别于上面的新评估。
"$P" -B "$ARTIFACT_ROOT/scripts/replay_release.py" \
  --configuration early --limit 1 --output-dir "$OUT/replay-early"
# 正确性：100%/空Skip、独立cache decode、清理及受控异常。
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" \
  --mode correctness --output-dir "$OUT/correctness"
```

输入：原包固定数据、绑定、模型；输出：独立新目录的逐题比较/`correctness.json`与完成/失败记录。以上引用旧v3保存的实测范围，本次未重跑。100%Encoder比较Encoder/postnorm/Connector/prefill logits及完整生成IDs；未独立比较Encoder decode logits。空Skip另含独立cache一步decode；二者不能代替真实50%/Multi的矩阵或非空Skip零调用证据。

## 7 性能profiler与成本

```bash
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" --mode performance \
  --configuration baseline --phase fixed16 --output-dir "$OUT/perf-baseline-fixed16"
"$P" -B "$ARTIFACT_ROOT/scripts/release_checks.py" \
  --mode profiler --output-dir "$OUT/profiler"
"$P" -B "$ARTIFACT_ROOT/scripts/analyze_costs.py" \
  --probe-root "$ARTIFACT_ROOT/evidence/day2_evening/cost_perf_v1" --output-dir "$OUT/costs"
```

输入：科学包配置/诊断样本和模型（成本只读旧原子记录）；输出：新目录中的measurements/hardware/binding、profiler trace或成本CSV/摘要。本次不执行。性能每次两集各1诊断题、2预热5正式；要比较六配置须同一运行版本成组并保留首尾baseline锚点。同步计时覆盖GPU输入到native generate返回，不含加载/Processor/H2D/评分/I/O，生产选择/回填/同步保留。

`natural`允许G变化；`fixed16`控制G=16，固定输出文本不进入质量表。`first/stages`是CLI可用但v3未实跑的范围；旧cost-perf-v1有保存结果。CPU成本估算不等于硬件FLOPs实测；模块独立median不相加减推因果。

## 8 已验证范围与未验证项

|版本/用途|已有实际范围|不能据此声明|
|---|---|---|
|原context-v2质量|六配置×两集×256=3072，保留逐题原子输出|本次重跑、新24加入主表|
|旧cost-perf-v1性能|780正式/312预热；图表主性能使用此版|与v3毫秒混合，自然G变短即同工作量提速|
|postday3-v3迁移|120正式/48预热，锚点另计；已有正确性/短profiler|冷安装、跨机或v3 first/stages实跑|
|新增薄入口|同机热缓存24条、partial隔离|新入口全256或新规范性能|
|本次仓库整理|只读结构/链接、CLI/help与定向文件保护检查|新增模型前向、GPU性能或科学正确性实验|

冷依赖安装、冷模型/数据获取、跨机、Windows/macOS/其他GPU、新入口完整256、Flash/SDPA、v3 first/stages及任意输入/OOM/kill仍未验证。报告负结果和AI/个人掌握边界不因导航整理改变。[证据索引](EVIDENCE_INDEX.zh.md)给出各原件位置；[当前发布页](PUBLICATION.zh.md)解释历史快照及现有Actions的用途。
