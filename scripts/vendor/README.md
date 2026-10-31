# 上游复用

`mmf_m4c_evaluators.py` 完整原样复制自 MMF，未修改评分公式或其他代码。

- 上游：https://github.com/facebookresearch/mmf/blob/72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py
- 使用：`EvalAIAnswerProcessor`、`TextVQAAccuracyEvaluator`。
- 许可证：同目录 `MMF_LICENSE`（BSD），保留文件版权声明。
- 未安装 MMF 框架；未使用文件中的其他评分器。

ScienceQA 指标定义来源：

- https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/tools/evaluate_acc.py
- 许可证：同目录 `SCIENCEQA_LICENSE`（MIT）。
- 原始文件完整保存在 `evidence/day1_morning/sources/scienceqa_evaluate_acc.py`。
- 本地 `check_scores.scienceqa_accuracy` 复用预测索引与标签索引相等的定义，
  将分母从上游固定的 4241 条 test 结果改为实际诊断子集长度；返回 0–1，
  上游显示为百分数。未复制其分组统计及 pandas 命令行入口。
- `extract_choice` 是本地必要适配，独立于指标定义；不读取参考答案，解析失败计错。

验证命令：`.venv/bin/python scripts/check_scores.py`。
结果以 `evidence/day1_morning/scoring_checks.json` 为准。
