# 当前发布状态与版本

**状态：公开仓库和正式 Release 已发布。当前阅读入口是 [main 首页](../README.md)，接收者命令以 [REPRODUCTION](REPRODUCTION.zh.md) 为准。** 本页是当前版本/发布说明；旧回执仍保留其记录时点的事实，不作为当前发布失败状态。

## 下载与版本

|载体|用途与关系|
|---|---|
|[当前 main](https://github.com/Sean-xzx/Smolvlm-Compression-Study/tree/main)|源码、报告入口、题目检查导航与当前说明；在初次交付之后继续更新呈现层和工装防误用|
|[正式 Release / tag](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/tag/autonomous-20261031-release)|`autonomous-20261031-release` 固定指向 `c8c31283aa066f5a0dc9036e7b9d56e93bde1386`；14项上传附件，另有GitHub自动源码包|
|[完整科学包](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/Shen-Wu-Challenge-postday3-v3.tar.gz)|postday3-v3原包；455733348 bytes；运行与全部原始证据的必要父依赖|
|[新增24验证包](https://github.com/Sean-xzx/Smolvlm-Compression-Study/releases/download/autonomous-20261031-release/new24-verification-20261031.tar.gz)|薄入口的小样本新预测，与原包/旧主实验分开保存|
|[新版PDF直接下载](https://raw.githubusercontent.com/Sean-xzx/Smolvlm-Compression-Study/main/report/REPORT.zh.pdf)、[Word直接下载](https://raw.githubusercontent.com/Sean-xzx/Smolvlm-Compression-Study/main/report/REPORT.zh.docx)、[Markdown](../report/REPORT.zh.md)|当前main为professor-report-v2：任务A/B/C正文与同报告附录，MD母稿→Word→PDF；旧Release报告不替换。slides仍为旧版|

科学原包 SHA256：

```text
d95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d
```

Release 中的 `Smolvlm-Compression-Study-Autonomous-20261031.zip` / `.bundle` 是发布时的代码快照，**不包含此后 main 的所有说明更新**。当前 main 与固定tag不能称为字节一致；此轮不移动tag、不替换或删除任何旧附件。按当前命令复现时，使用当前main薄入口配合原科学包，而不是把GitHub自动源码包当作完整实验包。

## 新版报告与旧快照

新版报告 professor-report-v2 重组旧17节，新增同报告附录并改进六张图；3072质量、780主性能及v3迁移范围不变。当前下载应使用上表 main 的直接链接。旧Release中同名REPORT附件、旧代码ZIP/bundle和talk均保留原字节，不能用旧附件SHA验证新报告。`verification/REPORT_RENDER_QA.json`、`FINAL_ACCEPTANCE.json`与`REPOSITORY_NAVIGATION_QA.json`均为之前版本的历史回执；新版使用独立[报告核版记录](../verification/REPORT_REFACTOR_QA.json)，实际发布后下载记录另见交接。

报告生成：编辑 `report/REPORT.zh.md` 母稿后运行 `scripts/build_report.py` 导出Word，再用documents的 `render_docx.py` 同源渲染PDF；生成器不会恢复旧17节。图表用独立文档依赖运行 `build_figures.py --artifact-root U`，只读原科学记录；不能改实验环境。

## 历史快照如何读

|保存的材料|正确解释|当前入口|
|---|---|---|
|Release内 `TASK_STATUS.json`、`AUTONOMOUS_HANDOFF.zh.md`|上传时历史快照，其中“待上传/待下载核验”等描述保留原时间与原结论|本页与[附件下载回执](../verification/PUBLIC_DOWNLOAD_RECEIPT.json)|
|[FINAL_ACCEPTANCE.json](../verification/FINAL_ACCEPTANCE.json)|发布前本地交付验收；其 `not_verified` 的公开发布阻塞是当时事实，未改写为本轮实验通过|本页；该JSON不是接收者自检输出|
|[上轮交接](FINAL_HANDOFF.zh.md)、[上轮阶段状态](FINAL_TASK_STATUS.json)|首次公开交付结束时的过程记录，含原机器路径；不取代当前README/复现页|[首页](../README.md)、[复现](REPRODUCTION.zh.md)|
|`REPRODUCTION.final.zh.md`、`REPRODUCTION.v2-history.zh.md`及v3适配记录|封存科学交付阶段的历史说明，不是另一个最新接收者入口|[REPRODUCTION.zh.md](REPRODUCTION.zh.md)|

当前[67项映射](REQUIREMENTS_MAPPING.zh.md)的S1与[E8附件入口](EVIDENCE_INDEX.zh.md#e8)反映已发布事实。原验收回执、科学proof与命令起止记录保持不动。

## 已有发布验证与范围

已有[源码回执](../verification/PUBLIC_REPOSITORY_RECEIPT.json)记录：本机匿名下载 `c8c3128` 源码ZIP，148文件匹配，报告PDF直接下载字节一致。[附件回执](../verification/PUBLIC_DOWNLOAD_RECEIPT.json)记录：GitHub托管Ubuntu无token/cookie/Authorization的完整下载，14项大小/SHA均通过。它们验证的是指定版本，**不是后来每次main改动的科学验收**。本轮[仓库导航检查](../verification/REPOSITORY_NAVIGATION_QA.json)只覆盖定向链接/结构、CLI/help与工装防误用，不重下载大附件、不追加GPU实验。

发布经Edge正常账号建库、已认证连接器推送、[Actions上传](https://github.com/Sean-xzx/Smolvlm-Compression-Study/actions/runs/41201440762)和[匿名下载核验](https://github.com/Sean-xzx/Smolvlm-Compression-Study/actions/runs/41202446542)完成。此前CLI/浏览器上传限制已不阻塞现有公开成果，未索取聊天密码或改变账号权限。[临时传输关闭记录](../verification/TRANSFER_SHUTDOWN.json)保留。

`.github/workflows/` 的两项工作流只用于上述一次性发布/下载核验，**不是科学测试CI**。当前取消push触发并将执行条件关闭，只保留原步骤供追溯，避免本次导航修改重传附件或覆盖历史回执。未来如需重发布，应另行审查版本与新传输入口。

## 本地工装与接收者检查

`scripts/validate_delivery.py` 是历史本机工装，依赖仓库外new24目录及旧ACCESS_BLOCKER；默认拒绝执行。需要研究历史逻辑时先看其 `--help`，显式历史模式也要求全新的仓库外输出文件，不会覆盖历史验收。一般读者使用 `python3 -B scripts/check_repository.py`，仅检查阅读材料/本地链接/配置；按需附 `--artifact-root` 或 `--archive`，缺少外部输入会明确显示未检查。它不证明科学正确性、GPU就绪、冷安装或推理成功。
