# 来源与本轮改动

完整上游固定commit、许可与v3适配历史保存在[封存来源文档原字节副本](SOURCES_AND_CHANGES.v3-history.zh.md)。其中“个人目标暂缓”等属于旧阶段状态，本轮已经完成talk目录准备材料，未代替用户个人能力验收。

|本轮新增|必要性与复用|实际验证|
|---|---|---|
|evaluate_release.py|旧重放强制历史预测相等，不能承担新评估；复用binding_for、context loader/build_question、patch_for、run_sample和summarize|真实六配置两集前2题=24预测；partial、分母、当前绑定与历史诊断；完整256未跑|
|build_figures.py|从旧质量原子记录、两成本CSV、逐重复性能和B4形状提取图源|3072质量/6144成本/24576层记录；780正式+312预热重汇总与旧摘要一致；无新性能|
|build_report.py|统一Markdown/Word正文与图表，Word导出PDF|17页逐页实际渲染检查；文档运行库兼容修复不改实验环境|
|build_mapping.py|逐项覆盖教授规格|67项、实际代码行/报告节/证据/边界；个人能力单列未验收|
|build_talk_content.py和build_talk.mjs|复用报告图制作12页可编辑演示及30分钟讲稿|结构/几何/字体与重新导入检查；逐页渲染，另导PDF。图表以报告图片嵌入，SVG/脚本另附|
|exercise_demo.py|低成本独立75%配置修改演示|原select_tokens/restore_grid的CPU fixture，K768、原序/回填通过；未做该新配置模型实验|

encoder_topk.py、ffn_skip.py、原加载/评分/统计helper、全部原配置和清单均按原字节复制，没有修改U、3072原子结果、旧780性能、v3 120性能、旧proof、原tar或实验.venv。完整比较见verification/FINAL_ACCEPTANCE.json。

来源：HF/Transformers原生模型/Processor/generate，MMF原样vendor，ScienceQA官方指标定义加本地解析；Projector四类仅文献/源码说明，本轮补清Flamingo固定学习latent的视觉Resampler。所有上游许可保留，见licenses及NOTICE。

AI实际完成主要适配、编排、统计、自动核查、正文、图表和汇报准备。用户提供目标和范围决策，未记录人工原稿/审阅/独立掌握。具体贡献与未验证范围见报告§17及交接。
