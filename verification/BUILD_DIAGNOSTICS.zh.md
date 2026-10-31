# 文档与发布工具实际记录

- 初始Matplotlib环境缺包：只在autonomous/build/plot-deps安装文档绘图依赖，实验.venv未改。
- 首次图源提取把stages分组字段误作phase，触发每组5重复断言；检查原始JSONL后改用stage字段，未改科学记录/数值。
- bundled LibreOffice/Poppler首次启动要求GLIBC_2.38/GLIBCXX_3.4.31，本机Ubuntu22.04为glibc2.35。仅从官方Ubuntu noble包目录获取libc6与libstdc++6并核包索引SHA，解包到build/lo-compat，以独立loader启动同一bundled soffice.bin；不替换系统/运行时/实验库。实际版本LibreOfficeDev26.8。Poppler使用系统版本。
- 报告首版19页含空白页和矩阵表溢出，修复标题border、段前分页及列宽后17页；最终全页检查与SHA在REPORT_RENDER_QA.json。
- PPTX初次finalizer缺RUNTIME_NODE_MODULES，保留候选并补正确运行时变量，最终结构/几何/字体/first-party重新导入通过。实际12页PNG及另行LibreOffice PDF逐页检查。
- 代码速查PDF初始Droid字体没有Latin字形；使用DejaVu Sans与Droid混排修复，最终一页已检查。
- 官方gh2.103.3下载SHA已核；实际auth status退出1：未登录。连接器profile/list可用但不提供建库/Release上传。浏览器现有凭据正常登录尝试返回用户名或密码错误，只尝试一次。

新24条的精确argv/cwd/开始结束/退出码和entry SHA完整保存在commands.json；早期文档制作尝试没有独立捕获精确开始时间，不补造。最终源与输出SHA、渲染回执和总验收具备独立记录。没有通过关闭assert或修改科学hash规避失败。
