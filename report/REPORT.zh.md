# SmolVLM无需训练压缩

视觉Token剪枝与FFN Skip的实验比较

Sean-xzx（GitHub账号） · 2026年10月31日

科学版本 postday3-v3 / context-v2 · AI辅助提交候选版

## 1 研究问题与主要发现

本研究在SmolVLM-256M-Instruct原始权重上，比较视觉Encoder Token Top-K与LLM FFN Skip两种无需训练的方法。六配置分别使用两个数据集的同一组256题，共12组3072条正式预测。压缩的价值分别由质量、计算量、实测速度和存储回答。

四种Encoder剪枝确实缩短当前FFN与后续Encoder的序列，并减少主要矩阵量；所有质量指标均下降。Late在四种Encoder配置中质量损失最小，Multi的质量损失最大。Encoder末尾回填规则网格适配原Connector，LLM视觉位置数保持不变。Skip省略三层FFN，但统一生成16 token时的旧主基准几乎没有收益。
![图1 真实计算路径。TextVQA row5有13个独立图块，T=889。示意依据encoder_topk.py、ffn_skip.py及B4实测形状，不是Attention归因图。](../figures/01_method.png)

图1 真实计算路径。TextVQA row5有13个独立图块，T=889。示意依据encoder_topk.py、ffn_skip.py及B4实测形状，不是Attention归因图。

六配置总参数256484928、原权重513028808 bytes均不变。没有证据显示显著allocated峰值节省。24条新增预测只验证接收者入口，不替换正式3072质量或旧780性能。本文由AI依据保存记录完成写作与自动核验，未记录Sean-xzx已审阅或已独立完成技术交流。

<!-- PAGE -->

## 2 实验协议与评分

固定模型为HuggingFaceTB/SmolVLM-256M-Instruct，revision为7e3e67edbbed1bf9888184d9df282b700a323964。TextVQA使用lmms-lab-encoder/textvqa@9c0699cd19768ac5ab97568f6b3cbac4c0062884；ScienceQA-IMG使用lmms-lab/ScienceQA-IMG@b9984af01798586705cee1fbdfa80a6dcbef04f1。

两集均为default/validation。各自独立用Random(20261029)从总体5000/2097条中无放回取256并升序，保存stable_id、row_index、图像SHA和清单meta。各配置输入prompt、input IDs及图片一致。图像以RGB无损PNG随包提供；Idefics3Processor执行原生resize、rescale、normalize和图像splitting，最长边2048，块512。正式图片每块1024视觉位置，无额外CLS。

Text输入图片、问题和短答案指令，不加入OCR字段。Science输入图片、非空原始hint、问题、原序选项和选项字母指令。answer仅在生成后评分，lecture/solution均不进入输入或评分。hint为合法题目背景，协议在控制诊断前冻结；旧无hint的58.984375%不进入本报告分差。

本机正式环境为Python3.10.12、torch2.7.1+cu128、Transformers4.57.6、RTX5090 Laptop 24463MiB、driver580.178.04。BF16、eager、batch1、eval/inference_mode、seed0、确定性、TF32关闭。pixel_values实际输入FP32，由原生模型内部按需转dtype。greedy、beam1、max_new_tokens64、原EOS/cache，六配置一致。

### 两个评分对象

Text使用原样MMF EvalAIAnswerProcessor及TextVQAAccuracyEvaluator。每题十参考答案，留一后对另外九条的匹配数m计算min(m/3,1)，再对十次留一平均，最后平均256题。若标准化后十条中有c次匹配，其闭式为：

```text
s(c) = ((10-c) min(c/3,1) + c min(max(c-1,0)/3,1)) / 10
c=0,1,2,3,4,10 对应 0,0.3,0.6,0.9,1,1
```

因此Text结果是软分，不能解释为整数答对率。历史1536条Text输出的闭式聚合差异为0；normalizer与MMF/Pythia同源，不能称完全异源评分验证。

Science指标采用官方选项索引准确率定义，实际生成文本由本地extract_choice v2保守、标签无关地解析。指标定义与文本解析是两个层次，未原样运行官方完整评测管线。空输出/解析失败保留原文、记0且保留256分母；推理异常保留错误并标未完成。旧正式3072条没有执行异常。证据E1/E5，完整协议见configs/eval_protocol_context_v2.json。

<!-- PAGE -->

## 3 Encoder Top K实现

记a=x+Attention(norm1(x))。选择发生在Attention残差之后、当前FFN之前。原生eager权重为[图块,head,query,key]，先用FP32对head求均值，再在有效query上平均，得到每个key的incoming attention分数。无效key置为负无穷。没有把key维误当query维，也没有只修改mask来冒充真实少算。

每块K=max(1,ceil(rate×当前有效token数))。按分数稳定降序取选集，平分时按原网格位置，随后恢复原始相对顺序。gather后的a进入当前norm2和MLP，输出为gather(a)+MLP(norm2(gather(a)))。当前层Attention仍使用剪前Na，当前FFN已使用剪后Nf，下一层Attention才变短。

### 六配置
|配置|Encoder层|每阶段率|实际K|最终率|Skip层|
|---|---|---|---|---|---|
|baseline|无|无|1024|100%|无|
|early|3|0.5|512|50%|无|
|middle|6|0.5|512|50%|无|
|late|9|0.5|512|50%|无|
|multi|3/6/9|0.5/0.5/0.5|512/256/128|12.5%|无|
|skip|无|无|1024|100%|8/16/24|

视觉Encoder共有12层，层号从1开始，L/4、L/2、3L/4在本模型直接对应3、6、9。Early、Middle、Late独立50%；Multi在同一次前向依次用各层当前序列重新打分，1024→512→256→128，最终12.5%。两类方法不叠加，Encoder实验保持LLM结构不变，Skip保持视觉分支不变。没有训练、比例/层扫描或最优组合搜索。

### 图块与padding

每个图像块作为独立batch项，块内Attention独立执行。Text5的13块不能合成长度13312的一条Attention序列。不同有效K用Kmax物理占位，padding不参与key、query评分和回填。正式24576层记录padding均0；人工有效数[16,5]的ceil测试依次得到[8,3]、[4,2]、[2,1]，只证明边界处理，不表示正式图像存在该padding。

保留token的原位置在Multi中逐层复合，供Encoder之后的规则网格适配使用。保留特征已经经前层Attention混合信息，删除某位置不等于把其全部信息从模型中抹除。真实50%路径、受控mask/残差、独立选集重建和恢复证据见E2/E3。

<!-- PAGE -->

## 4 逐层Token与真实少算边界
![图2 每块Attention输入Na和FFN输入Nf。来自Text5 B4/B6实测，baseline与Skip视觉路径相同；层号1起。](../figures/02_tokens.png)

图2 每块Attention输入Na和FFN输入Nf。来自Text5 B4/B6实测，baseline与Skip视觉路径相同；层号1起。
|层|baseline|early|middle|late|multi|
|---|---|---|---|---|---|
|1|1024/1024|1024/1024|1024/1024|1024/1024|1024/1024|
|2|1024/1024|1024/1024|1024/1024|1024/1024|1024/1024|
|3|1024/1024|1024/512|1024/1024|1024/1024|1024/512|
|4|1024/1024|512/512|1024/1024|1024/1024|512/512|
|5|1024/1024|512/512|1024/1024|1024/1024|512/512|
|6|1024/1024|512/512|1024/512|1024/1024|512/256|
|7|1024/1024|512/512|512/512|1024/1024|256/256|
|8|1024/1024|512/512|512/512|1024/1024|256/256|
|9|1024/1024|512/512|512/512|1024/512|256/128|
|10|1024/1024|512/512|512/512|512/512|128/128|
|11|1024/1024|512/512|512/512|512/512|128/128|
|12|1024/1024|512/512|512/512|512/512|128/128|

表内每格为Na/Nf，单位token/块。单点剪枝当层的Na仍为1024，Nf已为512；Multi第6层是512/256，第9层是256/128。完整两集逐层记录见E3。Encoder出口的稀疏长度分别为1024/512/128，后续回填只发生在Encoder之后，不将丢弃token送回后层Encoder。

<!-- PAGE -->

## 5 Projector与FFN Skip

### 四类连接方式
|模块|特征映射|对token数量N的典型作用|
|---|---|---|
|Linear|逐位置线性映射特征宽度|普通pointwise形式保持N|
|MLP|多层Linear与非线性|普通pointwise形式通常保持N|
|Q-Former|学习query通过视觉交互汇聚特征|输出由设计的query数决定|
|Perceiver Resampler|固定学习latent作为query，跨注意力重采样视觉特征|输出等于latent数，与输入网格长度解耦|

Perceiver Resampler具体对应Flamingo §3.1.1：学习latent query从可变长视觉特征中读入信息，典型输出64个视觉token；其key/value还拼接视觉特征与latent特征。这比泛称Perceiver IO输出query更贴合视觉Resampler。本研究只作概念说明，没有实现或训练额外连接器。[Flamingo原论文](https://arxiv.org/abs/2204.14198)；Q-Former见[BLIP-2](https://arxiv.org/abs/2301.12597)。

### 本模型的真实Connector

SimpleMLP类实际只有一个无bias Linear。1024→64的数量变化来自前面的4×4 pixel shuffle空间合并，通道768→12288，再Linear到576；不能由类名认定这里有多层MLP。剪枝分支先对保留特征执行原post_layernorm，再按原位置scatter到零网格；零填在postnorm之后，避免其bias把占位零变为非零。

Text5基线和剪枝分支最终均以[13,1024,768]规则网格进入Connector，pixel shuffle得到[13,64,12288]，Linear输出[13,64,576]。LLM视觉位置仍13×64=832，总prompt T=889。因此本适配只减少Encoder内部计算，未缩短原Connector或LLM的视觉序列。形状见E3/E4及冻结HF源码。

### Skip的残差与参数

在30层LLM的8、16、24层，同时省略post_attention_layernorm（norm2）和MLP，直接返回a=x+Attention(inputnorm(x))。inputnorm、Attention、cache保持。不是把MLP设Identity后再叠加norm2(a)，也不是先计算再乘零。三层为预先分散覆盖，未验证最优。

每层无bias gated FFN有3×576×1536=2654208参数，三层共7962624。另省三份norm2计算，权重1728参数；两部分均仍在state_dict中。总参数用sum(p.numel())统计，六配置均256484928；原权重stat为513028808 bytes，未生成更小checkpoint。停用计算、删除参数和文件压缩是不同量。调用计数及投影缺席见E4。

<!-- PAGE -->

## 6 正式质量结果
![图3 context-v2正式质量，六配置各两集256题。纵轴从0起；Text为软分，Science为准确率。全部值从3072原子记录重新汇总。](../figures/03_quality.png)

图3 context-v2正式质量，六配置各两集256题。纵轴从0起；Text为软分，Science为准确率。全部值从3072原子记录重新汇总。
|配置|Text %|Δpp|Science %|Δpp|Sci解析失败|截断T/S|
|---|---|---|---|---|---|---|
|baseline|54.4922|+0.0000|64.0625|+0.0000|1|0/1|
|early|43.5547|-10.9375|55.0781|-8.9844|1|0/0|
|middle|42.1484|-12.3438|58.2031|-5.8594|1|0/0|
|late|47.9688|-6.5234|60.5469|-3.5156|1|2/0|
|multi|13.6328|-40.8594|32.8125|-31.2500|28|5/2|
|skip|32.9688|-21.5234|32.4219|-31.6406|41|2/2|

12组均完整，无异常、缺失或重复。所有分差都相对同版context-v2 baseline。Late的质量损失在四种Encoder中最小，不能据此称其效率或其他分布最优。Multi和Skip的负结果保留，不为提高分数改解析器、删样本或重新选层。

Skip Science有41条解析失败，即使这些全部救回，上界也只有48.4375%，仍低于baseline64.0625%；因此格式失败不能解释全部损失。这个上界不排除未标记的其他解析问题。Science baseline64.0625%不等于官方模型卡73.6%的完整recipe复现；当前固定子集、提示和历史软件条件不同，不能直接把差距归因单一因素。证据E1/E5。

<!-- PAGE -->

## 7 Encoder与全模型计算量
![图4 主要矩阵/卷积FLOPs相对同版baseline的比例。每集256题离线均值，乘加计2；Encoder本体不含patch，G*=16为统一工作量估算。](../figures/04_cost.png)

图4 主要矩阵/卷积FLOPs相对同版baseline的比例。每集256题离线均值，乘加计2；Encoder本体不含patch，G*=16为统一工作量估算。
|配置|Enc Text GF|Enc Sci GF|Enc比|全Text GF|全Sci GF|全比T/S|
|---|---|---|---|---|---|---|
|baseline|2903.33|2986.38|1.0000|3195.75|3309.64|1.0000/1.0000|
|early|1649.62|1696.81|0.5682|1942.04|2020.07|0.6077/0.6104|
|middle|2045.53|2104.04|0.7045|2337.95|2427.30|0.7316/0.7334|
|late|2441.44|2511.27|0.8409|2733.86|2834.53|0.8555/0.8564|
|multi|1173.29|1206.85|0.4041|1465.71|1530.11|0.4586/0.4623|
|skip|2903.33|2986.38|1.0000|3180.68|3293.32|0.9953/0.9951|

GF单位为单题均值十亿FLOPs。Early、Middle、Late的Encoder本体分别减少43.182%、29.545%、15.909%；Multi减少59.588%。不能把最终保留50%或12.5%直接乘全Encoder或全模型。Skip保持视觉计算，G*=16全模型只省约0.47%/0.49%，视觉分支在这批输入占主要计算量。

G*=16是使用真实T、图块数和模型维度的离线统一长度估算，没有为全部512题另跑固定16性能或质量。实际G成本与此分开保存，各3072行。辅助打分、排序、gather/scatter、同步和移动不被称为免费。证据E6，完整均值/范围及每题表均保留。

<!-- PAGE -->

## 8 计算公式与实际生成长度

记图块B、初始N0=1024、视觉dv=768、mv=3072；LLM d=576、m=1536、30层、9Q/3KV、head64，d_kv=192、词表V=49280。Na/Nf按第4节逐层使用。主要矩阵/卷积计数如下，乘加2：

```text
patch = 2 B N0 (3 × 16²) dv
vision_layer = 8 B Na dv² + 4 B Na² dv + 4 B Nf dv mv
Connector = 2 B × 64 × 12288 × 576
LLM_layer = 4 q d(d+d_kv) + 4 q s d + 6 q d m
head_each_call = 2 × 1 × d × V
prefill: q=s=T ; decode j=1..G-1: q=1, s=T+j
Skip减少 = 3 × 6 × 576 × 1536 × (T+G-1)
```

首输出token归prefill，后G−1步为decode，G=1时没有decode；视觉和Connector只在prefill执行一次，输出头每次只处理最后位置。GQA的K/V投影宽192，Attention按query总宽576计。总量逐项相加，Skip的FFN层数为27，Attention仍30。

### 实际G的全256题分项均值
|任务|配置|视觉含patch|Connector|prefill|decode|head|合计GF|
|---|---|---|---|---|---|---|---|
|textvqa|baseline|2919.83|12.37|258.48|1.18|0.30|3192.16|
|textvqa|early|1666.12|12.37|258.48|1.19|0.30|1938.46|
|textvqa|middle|2062.02|12.37|258.48|1.14|0.29|2334.32|
|textvqa|late|2457.93|12.37|258.48|1.32|0.33|2730.43|
|textvqa|multi|1189.79|12.37|258.48|1.47|0.36|1462.48|
|textvqa|skip|2919.83|12.37|243.65|1.28|0.34|3177.47|
|scienceqa|baseline|3003.35|12.73|288.42|0.98|0.26|3305.73|
|scienceqa|early|1713.77|12.73|288.42|0.82|0.22|2015.96|
|scienceqa|middle|2121.01|12.73|288.42|0.83|0.22|2423.20|
|scienceqa|late|2528.24|12.73|288.42|0.91|0.24|2830.54|
|scienceqa|multi|1223.82|12.73|288.42|0.88|0.23|1526.08|
|scienceqa|skip|3003.35|12.73|272.34|0.98|0.27|3289.65|

FP32 score的基本算术按B×N²×(H+2)单列，cast按BF16读2+FP32写4 bytes/元素；每stage每块3次稳定argsort，另列gather/scatter与逐块CPU同步。bias、激活、RMS/softmax部分基本运算在成本CSV单列。rope、查表、比较排序、KV移动、预处理/H2D/I/O等未并入主要矩阵总数。它是约定估算，不是逐硬件指令总数或硬件测得FLOPs。

<!-- PAGE -->

## 9 自然生成性能主基准
![图5 cost-perf-v1自然生成。每集按长度选4题，每题2预热5正式；纵轴为逐题median的算术均值。图内G为对应4题平均实际生成长度。](../figures/06_natural.png)

图5 cost-perf-v1自然生成。每集按长度选4题，每题2预热5正式；纵轴为逐题median的算术均值。图内G为对应4题平均实际生成长度。
|任务|配置|wall ms|加速比|平均G|峰值MiB|
|---|---|---|---|---|---|
|textvqa|baseline|297.235|1.000|6.5|2901.84|
|textvqa|early|193.208|1.538|6|2901.99|
|textvqa|middle|221.912|1.339|5.5|2901.99|
|textvqa|late|248.093|1.198|5.25|2901.99|
|textvqa|multi|167.404|1.776|5|2901.99|
|textvqa|skip|257.427|1.155|3.5|2901.84|
|scienceqa|baseline|271.576|1.000|3.5|2902.27|
|scienceqa|early|169.816|1.599|4|2902.42|
|scienceqa|middle|213.897|1.270|4|2902.42|
|scienceqa|late|249.613|1.088|4|2902.42|
|scienceqa|multi|146.713|1.851|3.5|2902.42|
|scienceqa|skip|253.235|1.072|3|2902.27|

计时为GPU上已有输入到native generate返回并同步，主机perf_counter wall为主。不含模型加载、图像解码、Processor、H2D、评分、I/O和外部调试hook；生产score/sort/gather/scatter/同步保留。CUDA Event是辅助stream elapsed，不是纯kernel时间求和。自然G不同，Text Skip从6.5降到3.5，不能把自然速度改善全归因三层FFN少算。约2902MiB绝对allocated峰值几乎不变，剪枝略增，没有显示显著省显存。

<!-- PAGE -->

## 10 固定长度与阶段测量
![图6 cost-perf-v1固定G=16，每集1诊断题、2预热5正式。柱为median，误差线为IQR，不是总体置信区间。](../figures/05_fixed16.png)

图6 cost-perf-v1固定G=16，每集1诊断题、2预热5正式。柱为median，误差线为IQR，不是总体置信区间。
|任务|配置|首token ms|固定16 ms|固定16加速比|
|---|---|---|---|---|
|textvqa|baseline|219.35|414.22|1.000|
|textvqa|early|118.47|338.77|1.223|
|textvqa|middle|156.62|391.04|1.059|
|textvqa|late|182.99|401.35|1.032|
|textvqa|multi|104.65|323.14|1.282|
|textvqa|skip|221.71|417.23|0.993|
|scienceqa|baseline|236.52|417.66|1.000|
|scienceqa|early|129.78|364.02|1.147|
|scienceqa|middle|169.30|374.21|1.116|
|scienceqa|late|193.59|412.21|1.013|
|scienceqa|multi|110.84|325.15|1.285|
|scienceqa|skip|235.76|416.89|1.002|

首token为每集4题的逐题median均值，fixed16每集仅1题。Skip固定16为0.993×/1.002×，未显示明确同长度收益；Early为1.223×/1.147×，Multi为1.282×/1.285×，均伴质量损失。fixed16用min=max_new_tokens16改变停止行为，文本不进正式质量表。

旧主基准共780正式/312预热：natural240、first240、fixed16 60、stages/decode240；首尾锚点另20正式。锚点变化Text−2.46%、Science+3.83%，未过10%线。没有锁频，小n和硬件波动仍限制外推。独立模块median不能相加为端到端，也不能用总wall减vision推出LLM耗时。单步decode每次用fresh prefill cache和固定baseline首token，wrapper含位置/cache张量构造，不是纯kernel。

<!-- PAGE -->

## 11 正确性证据分别证明什么
|检查对象|保存证据与比较范围|结果与边界|
|---|---|---|
|单点100%及Multi全100%|两输入×四模式，Encoder/postnorm/Connector/prefill logits；完整生成IDs|32张量比较最大误差0，atol/rtol=0；未单独比较Encoder decode logits|
|空Skip|两输入prefill与独立cache一步decode最后位置logits，完整IDs|最大误差0，atol/rtol=0；cache独立|
|关闭额外路径|100%原生旁路|score/scatter调用0，只证明无压缩一致性|
|真实50%/Multi|四配置各4真实样本、受控mask/残差与选集重建|当前FFN及后层真实短；不由100%旁路替代|
|padding与原序|有效[16,5]受控输入；24真实stage独立重建1D评分选集|禁用key权重109584项最大0；扰动padding不改有效输出|
|grid适配|postnorm后保留位置比较及丢弃位置|保留相等、丢弃精确0；原Connector复用|
|清理|全named_modules forward/hook集合、参数身份/指针/version|正常退出及五配置受控异常恢复；任意OOM/kill未验证|

以上为旧context-v2及v3保存的实测证明，本轮只读提取，没有重新跑全部proof。100%走原生旁路，所以它不检验50%分数排序或gather的实现；真实剪枝由独立受控数值、pre-hook与矩阵证据支撑。有限固定样本通过不等于任意输入形式的数学保证。

### Skip真实未执行

旧skip_trace在基线与Skip同输入prefill和一步decode分别计数：目标L8/16/24的norm2和MLP为0，Attention/inputnorm各1次每阶段；gate/up/down目标CPU区间缺席，其他层正常。v3短trace另见MLP投影矩阵180→162，三层×三投影×两阶段共减少18。零调用与矩阵缺席结合说明没有先算再丢弃。不能把CPU marker时间与嵌套GPU事件重复相加。直接索引E2/E3/E4/E7。

### 存储与显存的证据边界

所有权重仍注册，因此参数停用不是checkpoint缩小。allocated峰值几乎相同，reserved是分配器历史，不能代替活跃显存或权重大小。早期仍为1024的密集Attention可能决定峰值，此解释与形状相容，但没有独立内存分配时间线因果定位，不能写成已证明原因。

<!-- PAGE -->

## 12 当前FFN与后层矩阵证据

固定Text5、T=889、B=13，单次诊断prefill。下表为fc1的pre-hook及aten::addmm输入，证明当层FFN及下一层已收到短张量；完整fc1/fc2见figures/matrix_evidence.csv及E3 JSON。13块可在Linear中展平BN行，Attention仍为B×N²。
|配置|层|pre-hook形状|二维矩阵输入|CPU us|CUDA us|
|---|---|---|---|---|---|
|baseline|3|[13, 1024, 768]|[[13312, 768], [768, 3072]]|31.26|986.68|
|baseline|6|[13, 1024, 768]|[[13312, 768], [768, 3072]]|31.52|1047.13|
|baseline|9|[13, 1024, 768]|[[13312, 768], [768, 3072]]|30.63|1051.00|
|early|3|[13, 512, 768]|[[6656, 768], [768, 3072]]|49.42|450.59|
|early|4|[13, 512, 768]|[[6656, 768], [768, 3072]]|31.16|449.24|
|middle|6|[13, 512, 768]|[[6656, 768], [768, 3072]]|68.00|527.58|
|middle|7|[13, 512, 768]|[[6656, 768], [768, 3072]]|32.56|526.72|
|late|9|[13, 512, 768]|[[6656, 768], [768, 3072]]|105.30|446.30|
|late|10|[13, 512, 768]|[[6656, 768], [768, 3072]]|48.64|443.07|
|multi|3|[13, 512, 768]|[[6656, 768], [768, 3072]]|110.10|458.17|
|multi|4|[13, 512, 768]|[[6656, 768], [768, 3072]]|50.16|455.80|
|multi|6|[13, 256, 768]|[[3328, 768], [768, 3072]]|46.99|209.70|
|multi|7|[13, 256, 768]|[[3328, 768], [768, 3072]]|32.46|205.31|
|multi|9|[13, 128, 768]|[[1664, 768], [768, 3072]]|46.56|109.57|
|multi|10|[13, 128, 768]|[[1664, 768], [768, 3072]]|43.11|106.62|

CPU us为该算子主机总区间，CUDA us为关联设备时间。单次profiler用于检查执行路径和矩阵shape/time，不能替代规范延迟。Multi在L3、L6、L9分别用512、256、128，L4、L7、L10也持续缩短；剪枝当层Attention仍是旧Na。fc2相应输入为[13,K,3072]，矩阵为[13K,3072]×[3072,768]。

每配置完整12层QKV/O、QK/AV、fc1/fc2，以及score/select/scatter的具名区间均在B4各配置JSON/trace；v3独立短profiler再次覆盖Multi三个stage及Skip目标。旧证据引用与本轮实际新24条分开，不把文件存在称为本轮重新执行通过。

<!-- PAGE -->

## 13 代表题与子集范围

TextVQA row5的baseline原始输出为“22.”，Multi为“2.”，得分1→0。该案例按照预定类别的最小原索引选择，不是事后挑选最好看的样例。原图如下；没有生成虚假的Attention热图。

![TextVQA row5原图](../figures/text5.png)

图7 TextVQA固定row5原图，来自随包RGB PNG。baseline与Multi使用同图同prompt；该题13独立图块，LLM视觉位置均832。原图/问题/参考答案/原始输出以原子JSON为准，版权归原图权利人。

Science row165的baseline“Answer: C”、Late“Answer: A”，得分1→0；row42 baseline“Answer: C”、Multi“Answer: B”，得分1→0。row60 Skip输出含互相矛盾的表达，保守解析拒绝。完整raw、参考和类别规则见E5 failure_cases.json。

这些记录说明固定输入下计算改变伴随质量和格式损害；未做Attention归因实验，不能声称已定位被删文字patch，或证明Multi因某个具体对象被删除而答错。

### 基线256题实际长度与成本范围
|任务|图块B范围|输入T范围|实际G范围|总GF范围|
|---|---|---|---|---|
|textvqa|9–17|622–1163|3–16|2092.5–3991.7|
|scienceqa|5–17|379–1394|2–64|1164.3–4081.6|

每题输入/输出长度、全部配置实际G与固定G16成本均有原表；figures/ranges.csv保留每组min/max，quality.csv保留G均值/min/max和截断数。性能题按长度选取，不是对全部512题延迟总体的随机推断；5次重复也不是5个独立任务。

<!-- PAGE -->

## 14 接收者新评估入口验证

旧evaluate_context.py完成过正式评估，但有历史环境/证明门禁；replay_release.py能迁移路径，却强制预测与历史逐字一致且没有完整汇总。因此本轮增加evaluate_release.py薄入口，显式读取artifact-root，复用原加载、问题构建、补丁、generate、评分、原子保存和summarize，不复制模型或另造评分框架。

新运行保留模型revision、科学源/配置/manifest/协议SHA和必要依赖检查，当前Python二进制/GPU环境另行绑定。历史环境ID只是引用，不伪造为本轮环境。逐题保存新生成raw/new IDs、prompt/input IDs、分数、解析、T/G、错误和时间。历史相等只输出诊断，生成差异不直接等于实现失败。
|配置|Text索引|Science索引|预测数|partial|本机历史诊断|
|---|---|---|---|---|---|
|baseline|5,19|2,9|4|是|全部相同|
|early|5,19|2,9|4|是|全部相同|
|middle|5,19|2,9|4|是|全部相同|
|late|5,19|2,9|4|是|全部相同|
|multi|5,19|2,9|4|是|全部相同|
|skip|5,19|2,9|4|是|全部相同|

2026年10月31日真实串行执行六进程，每配置加载模型一次、处理两集各前2题，共24条。六命令exit0，ID正确、24条完整、无异常或解析失败；prompt/input IDs/shapes/raw/new IDs/score/parse/T/G均与本机历史相同。partial_mean_score不写成全256分数，正式mean_score为空，原3072主表保持不变。精确argv/cwd/开始结束/退出码/日志/SHA见verification/commands.json与NEW24_ACCEPTANCE.json。

### 运行职责

接收者先准备独立Python依赖和固定模型/Processor/tokenizer缓存，再获取完整包、核SHA、解包。数据就绪检查核对随包512PNG，不是数据下载器。新预测评估用本轮入口；严格精确重放、正确性、性能/profiler和CPU成本沿用完整包入口。所有输出位于包外新空目录。

可直接复制的完整命令见docs/REPRODUCTION.zh.md。轻量Git不含全部父证据和图片，必须与完整科学包组合；命令不需要作者机器的固定路径。新入口limit256、冷安装、跨机器尚未实测；本轮不为文档重跑主实验。

<!-- PAGE -->

## 15 独立解包验证与版本隔离

postday3-v3是旧科学资产的迁移功能验证。保存回执记录独立解包后27条命令exit0，512PNG SHA通过，100%/空Skip比较最大误差0，六配置各两集首题12条严格重放对齐，CPU两套3072行成本重算字节相同。最终包与已测RC的执行/输入文件SHA一致，文稿修改后没有重复GPU。

v3性能每集1诊断题，六配置natural/fixed16各5正式、2预热，共120正式48预热；首尾锚点另20/8。下面独立列出v3固定16，不与旧780毫秒或baseline拼接。
|任务|配置|fixed16 median ms|IQR ms|同版加速比|
|---|---|---|---|---|
|textvqa|baseline|417.908|15.144|1.000|
|textvqa|early|340.159|22.275|1.229|
|textvqa|middle|372.487|5.300|1.122|
|textvqa|late|409.724|37.793|1.020|
|textvqa|multi|328.193|10.412|1.273|
|textvqa|skip|410.736|2.944|1.017|
|scienceqa|baseline|428.611|10.214|1.000|
|scienceqa|early|344.996|11.775|1.242|
|scienceqa|middle|371.340|1.314|1.154|
|scienceqa|late|412.326|8.225|1.039|
|scienceqa|multi|329.655|12.849|1.300|
|scienceqa|skip|417.230|1.875|1.027|

v3锚点Text−1.5501%、Science−1.1601%，未触发10%线。小样本验证表明迁移路径在同机共享解释器/HF热缓存工作，不代表跨机冷启动或整个数据分布。v3 first/stages、严格replay limit256未实跑。旧780、v3 120、新24条是三套不同范围；新24条的调度时间也不是规范性能测量。包外FINAL_RECEIPT及E7保存完整边界。

原v3包455733348 bytes，SHA256 d95a7774653f0b35f5e7cb37619807c27b0d295e0494d29b90b4be22bcec0d4d。它不担保本轮新增文稿/入口；新增文件使用本轮清单和提交版本。公开发布的实际状态单独见docs/PUBLICATION.zh.md，不用本地文件冒充线上附件。

<!-- PAGE -->

## 16 讨论与适用边界

### 质量与计算的取舍

Late保留更多层的完整视觉交互，当前质量较好而成本节省较小；Early成本节省更多，质量下降更明显。Multi叠加三次选择，质量损失极大。这些是本固定协议下的观察，不能解释为已证最优位置或已确定质量损失的唯一机制。没有训练、消融扫描、语义归因或补偿模块实验。

### FLOPs与时间为什么不成比例

主要矩阵量没有包含稳定排序、gather、零回填、CPU同步和框架调度等全部开销。本实现显式取eager Attention权重，baseline也使用eager；Flash/SDPA后端未测，不能外推部署加速。旧独立Connector回放中剪枝约4–5ms、baseline约0.3–0.5ms，与适配开销有实际落点，但模块独立median不能相减或相加为端到端分解。

自然生成还受实际G影响。Skip减少三个FFN的矩阵量相对全模型很小，固定16几乎不变；自然少输出时wall缩短并不能证明同工作量提速。部分独立模块median顺序不符合包含关系，可能受时钟/调度/执行顺序影响，当前未通过配对实验定位具体原因。

### 外推与未验证范围

只有一个GPU、BF16、batch1、eager及两个固定validation子集。质量与离线成本每集256，规范性能自然/first每集4题、fixed16/stages每集1题、各5重复，未给总体置信区间。没有明显存储或allocated峰值下降证据。官方73.6%历史recipe未完整复现，Text标准化同源，Science文本解析是本地适配，均保留在结论边界中。

接收者新入口完整256、独立依赖安装、冷模型/数据下载、跨机运行未测试；v3 first/stages未运行；未重建全部历史proof。既有失败日志原样保存，不通过禁用assert、修改hash或参考结果消除差异。本轮未发现需要重做主科学结论的确切新缺陷。

### 可支持的结论

该实现真实缩短Encoder内部计算张量，并在本机小基准测得部分速度改善，代价是质量下降；它不减少LLM视觉序列、权重存储或显著allocated峰值。三层FFN Skip在当前选择上质量代价大，同长度收益不明确。这一负结果本身构成比较结论，不需要通过追分或隐藏配置改成正面结果。

<!-- PAGE -->

## 17 来源与实际贡献

模型/Processor/generate与Idefics3/Llama直接复用HF Transformers4.57.6；EncoderTopKPatch和SkipPatch是基于固定源码的实例级必要适配。Text评分文件原样复用MMF，Science准确率定义参考官方实现，文本解析为本地标签无关适配。成本/计时复用原项目helper，未复制完整训练或评测框架。

本轮AI完成接收者薄入口、24条真实推理编排与验收、数据只读提取、图表/图注、报告正文与排版、证据索引、汇报与练习准备。旧实验记录已披露AI主要代码适配、实验编排、统计与自动核验。Sean-xzx提供任务、约束、公开选择和范围决策；没有证据的人工原稿、逐题人工核实、审阅、练习或独立掌握不作声明。

已执行检查是自动检查：原子记录汇总、历史SHA绑定、新24条预测/摘要校对、真实渲染检查。报告/汇报材料是准备成果，不证明未来现场指定环节已由用户独立完成；该环节可查代码文档，应按教授要求不调用AI/Agent。未进行的用户审阅与个人能力验收分别记录于交接。

### 固定来源导航

1. [模型固定revision](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964)，Transformers4.57.6；源码SHA/许可见完整包evidence/day1_morning/sources及docs/SOURCES_AND_CHANGES.zh.md。
2. [TextVQA挑战评分说明](https://textvqa.org/challenge/)；[MMF固定评分源码](https://github.com/facebookresearch/mmf/blob/72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py)。聚合独立、normalizer同源，详细1536条检查见E5。
3. [ScienceQA官方准确率定义](https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/tools/evaluate_acc.py)。本地extract_choice源码与边界检查见E5。
4. [题目Token压缩参考](https://arxiv.org/abs/2504.01690v2)，仅方法参考，未运行音频框架；[LLaVA projector builder](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/multimodal_projector/builder.py)。
5. [BLIP-2 Q-Former](https://arxiv.org/abs/2301.12597v3)；[Flamingo Perceiver Resampler §3.1.1](https://arxiv.org/abs/2204.14198)。本轮补清固定学习latent语义，不引入新模型依赖或实验。

上游许可按文件保留，没有将混合工程统一声明MIT/Apache。图表数值全部取自figures/source_inputs.json登记的真实记录；保存CSV/JSON、绘图脚本、SVG/PNG和figure_manifest。原稿母版为本Markdown，同源导出Word再生成PDF，避免维护不同正文。

### 阅读和复现入口

[接收者命令页](../docs/REPRODUCTION.zh.md)、[证据索引E1至E8](../docs/EVIDENCE_INDEX.zh.md)、[教授逐项映射](../docs/REQUIREMENTS_MAPPING.zh.md)、[公开状态](../docs/PUBLICATION.zh.md)。完整源/图表与离线技术交流材料由README统一导航。
