# 三项小修改练习与参考解法

这些是AI准备的操作材料。已运行的只有下述独立CPU helper演示，用户独立练习与未来现场操作尚未发生。指定现场环节可查本页及代码，不调用AI/Agent。所有实验输出必须新目录，不能编辑冻结配置或修改hash绕过门禁。

## 练习1 将Early保留率从50改为75

任务：保持第3层选择，保留率改为0.75。先写出预期Na/Nf，再运行低成本helper检查ceil、原序和回填。不要先跑256题。

参考解法：新建独立exercise配置，记录parent配置路径/SHA，layers_1based=[3]、keep_rates=[0.75]，明确kind=exercise-only-not-frozen-protocol。不得改configs/context_v2/encoder_early.json或其正式binding。`evaluate_release.py`只支持冻结六配置，不接受新研究配置；不能通过改旧SHA来让它接受演练。

```bash
P=/path/to/supported/python
ARTIFACT_ROOT=/path/to/artifacts/Shen-Wu-Challenge
"$P" -B scripts/exercise_demo.py --artifact-root "$ARTIFACT_ROOT" \
  --output-dir /path/to/new-empty-exercise-output
```

实际由AI执行的独立CPU演示见 [exercise_demo.json](exercise_demo.json) 和 [exercise配置](exercise_early75.json)。原版select_tokens将[1,1024,768]真实gather为[1,768,768]，K=[768]，所选原位置256..1023升序；restore_grid返回[1,1024,768]，保留区逐值相同，丢弃区精确0；原冻结配置SHA前后一致。评分使用确定性递增fixture，与模型Attention分数无关。

如果将同改动用于未来真实模型，预期L1/2为1024/1024，L3为1024/768，L4–12为768/768；Text5若仍13块，fc1预期[13,768,768]，后Connector回填保持1024格、LLM仍832视觉位置。参数/原权重文件保持；Encoder理论成本高于50%剪枝、低于baseline。质量和速度方向未知。后者均为结构/公式推导，**没有跑这个新配置的模型前向、质量或性能**。

## 练习2 将Skip层列表改为5 15 25

任务：用独立新配置记录[5,15,25]，指出哪些函数与计数需检查，并解释为何不能预言质量。

参考解法：SkipPatch(model, [5,15,25], expected_source_sha256)在实例级安装；上下文退出恢复。真实模型使用前仍校验原生Llama源SHA、层范围/唯一性和eval路径。应保留inputnorm/Attention/cache，只省目标norm2+MLP，返回Attention残差a。不要把MLP替换Identity再保留第二残差。

预期目标层norm2/MLP零调用，其余正常，prefill+1decode的目标gate/up/down marker缺席；用同输入baseline作调用对照。三层数量和各层宽度相同，停用MLP仍7962624参数，norm2另1728；同T/G的主要矩阵减少公式仍15925248(T+G−1)，视觉/Connector/LLM输入长度不变，权重未删。质量、自然G、wall未知，不写成实测。本练习未运行新Skip模型。

如需真正评估新配置，另立版本和新binding，保留原六配置/源/证明，先最小正确性和路径检查，再明确新实验范围。本准备包不默认追加该研究实验。

## 练习3 从结论定位原子记录或重算成本

任务：找到baseline Text5的T/G、原始输出和分数；再找到Multi第6层fc1的pre-hook、矩阵与CPU/CUDA时间。可选将旧成本重算到新目录，不覆盖科学包。

参考解法：

```bash
"$P" - "$ARTIFACT_ROOT" <<'PY'
import json,sys
from pathlib import Path
u=Path(sys.argv[1])
r=json.loads((u/'results/context_v2_baseline_20261030/textvqa/samples/row_00005.json').read_text())
print({k:r[k] for k in ['stable_id','input_length','generated_length','raw_generated_text','score','parse_status']})
b=json.loads((u/'evidence/day2_afternoon/20261030_context_v2/b4/multi.json').read_text())
label='Vision.L06.mlp.fc1'
print(b['module_input_shapes'][label])
print([x for x in b['matrix_operators'] if x['marker']==label])
PY
# 可选CPU离线重算：输出必须新目录，不是新模型推理。
"$P" -B "$ARTIFACT_ROOT/scripts/analyze_costs.py" \
  --probe-root "$ARTIFACT_ROOT/evidence/day2_evening/cost_perf_v1" \
  --output-dir /path/to/new-empty-cost-output
```

正确定位预期：Text5 T=889、G=5、raw为“22.”（可含原前导空格）、score=1、parse=ok；Multi L6 fc1输入[13,256,768]，矩阵[3328,768]×[768,3072]，对应单次诊断CPU约46.992us、关联CUDA约209.695us。它是profiler诊断，不是稳定基准。以上值从真实记录读取并进入报告§12；本轮未为练习再执行整套旧成本重算。

评分标准：能从结论找到正确版本的代码/输入/证据，说明真实运行与推导边界，修改不污染冻结源，不把小样本当完整评估。此文件不代替个人能力验收。
