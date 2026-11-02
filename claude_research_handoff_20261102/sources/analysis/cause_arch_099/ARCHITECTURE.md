# M099：以事件定位状态为核心的定向语义读取架构草稿

日期：2026-10-20。**DESIGN_ONLY / UNVERIFIED / NOVELTY_UNCERTAIN**。H099未验证，M099未采纳。不是本轮要训练的模型。

继承硬事实：E091几何主配方停止；E097在旧F/H容量下拒绝、未训练；E098严重低读取仅47/334，原因前提拒绝、未训练。M091/H097/M097/H098/M098不升格，不复活旧合同。

## 1. 如果H099成立，架构应改变什么

原模型已有两个输出头，但它们共享上游事件query：

`同一query → 分类loss；同一query → 时间mask loss → 两种梯度共同更新该query及decoder`。

本草稿改变核心状态的训练职责：

`实体对时序特征 → 9个定位事件状态 → 时间mask/区间`

`定位事件状态及其预测支持（停止分类梯度回传）＋时序特征 → 同slot语义状态 → 谓词`。

含义：先让“这一次事件何时成立”的状态由定位监督维护；分类可以读取该状态及其所圈定的内容，但不能直接把定位状态推回只利于类别判断的方向。不同slot仍允许重叠事件，同一pair可多谓词并存。没有把视频切成互斥动作序列。

这不是说定位完全不需要语义，也不是把BIG原样颠倒为两套独立模型。训练中的联合事件匹配仍可使用类别与mask；类别通过匹配提供离散的事件对应信息，**不等于所有分类影响都被移除**。受控的是共享可微状态上的梯度通路。

## 2. 状态、信息流和输出

令冻结或共享视觉输入为f，定位decoder得到9个状态u：

\[
u_q=D_{\rm event}(f,q),\quad
m_q(t)=\sigma(w(u_q)^\top f_t).
\]

语义状态读取同slot的预测支持：

\[
r_q=\frac{\sum_t\mathrm{sg}(m_q(t))f_t}
          {\epsilon+\sum_t\mathrm{sg}(m_q(t))},\quad
v_q=D_{\rm semantic}\big(r_q,\mathrm{sg}(u_q),e_q\big),\quad
p_q=\mathrm{softmax}(C(v_q)).
\]

sg为停止梯度；e_q为语义槽标识，不增加输出query数。暂定用可学习小语义状态而非仅pooling，避免完全相同mask的不同关系必须得到完全相同类别；其是否足够仍未验证。

| 组成 | 产物 | 与假设的对应关系 |
|---|---|---|
| 事件定位状态u | 9条时间mask | 定位状态直接承接mask监督；不由分类loss直接更新 |
| 预测支持上的语义读取r/v | 每slot谓词分布 | 分类仍利用当前视频和预测事件内容；推理不用GT裁段 |
| 同slot事件匹配 | 类别＋时间成对输出 | 防止两支独立后类别与时间任意拼接；保留一对一事件监督 |

若训练编码器，分类梯度仍可经f间接污染定位输入。**首个架构实现应冻结f或明确切断该间接通路，不能只对u做stop-gradient就声称消除了所有共享干扰。** 若未来解冻，必须重新界定原因范围；本轮没有授权扩大。

原focal/Dice和类别目标可保留，不加CDF/W1、不改attention列预算、不通过多给query提高上限。对事件状态的职责划分是架构主张；具体参数量尚未定型，不能宣称容量公平或性能领先。

## 3. 为什么先做梯度合同，而不是直接训练双状态模型

[E099](EXPERIMENT_CONTRACT.md)在**相同前向与参数**下只改分类梯度方向，排除了更大模型、更多槽、新的局部上下文或新loss的解释。如果这一操作都不能形成完整的限域证据链，就没有理由用更复杂的双状态模型掩盖失败。

即使E099通过，也只支持为定位维护专门状态的动机。它保留有益分类梯度，而上述草稿阻断直接分类梯度更彻底，可能损失有益迁移；**E099阳性不等于M099已验证**。本轮不把架构训练追加进合同。

## 4. 文献对齐：新颖性不能建立在“分类定位有冲突”上

复用[091的26项基础调研](../../literature/short_interval_cause_arch_091_survey.md)及[097动态读出近邻](../../literature/short_interval_cause_arch_097_survey.md)。以下是本轮与H099直接相关的定向补充，访问日期2026-10-20。F=取得正文并核读所列关键节；A=仅摘要/官方页；不把取得全文等同逐字通读。

| 近邻与来源 | 本轮阅读 | 已有设计；对本草稿的限制 |
|---|---|---|
| [VrdONE，MM2024](https://arxiv.org/html/2408.09408v2) | F：联合类别/mask、训练部分；另核本地predictor源码 | 已经联合关系类别和实例mask，已有两个输出头与匹配。不能说其没有事件监督或首次将关系表示为事件。本草稿改变共享状态的梯度职责。 |
| [Gradient Surgery for Multi-Task Learning / PCGrad，NeurIPS2020](https://arxiv.org/html/2001.06782v4) | F：§2.2–2.3、算法1 | 已用梯度投影去冲突，并强调负夹角本身不足以解释有害优化。E099的投影不是新算法，必须归引；VidOR短事件的受控机制证据才是待建立部分。 |
| [Decoupling Localization and Classification in Single Shot Temporal Action Detection，2019预印本](https://arxiv.org/abs/1904.07442) | A：摘要；HTML正文未获取 | 时间动作检测早已有分类/定位私有分支。不能宣称首次解耦时间定位和分类；未凭摘要推测其梯度消融细节。 |
| [Rethinking Classification and Localization for Object Detection，CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Rethinking_Classification_and_Localization_for_Object_Detection_CVPR_2020_paper.html) | A：官方摘要及方法摘录 | Double-Head对分类/框定位使用不同头结构。两个head、任务特有表示均不是新意。图像框任务结果不证明本项目短关系原因。 |
| [TOOD，ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html) | A：官方摘要；PDF方法/分析摘录 | 处理分类定位不对齐，既保留任务交互又有特化。反例提醒：完全隔离任务未必优于联合。本草稿不能只用“冲突”论证解耦一定好。 |
| [Temporal Action Localization with Enhanced Instant Discriminability，2023扩展预印本](https://arxiv.org/html/2309.05590) | F：§3.5及引言，区别于TriDet的CVPR2023初版 | 为分类和定位安排不同特征金字塔，讨论时间可区分性。表示特化已有；这里不重新跑融合或把其rank-loss论证移植成VrdONE根因。 |
| [Temporal Action Localization with Cross Layer Task Decoupling and Refinement，AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/32516) | A：官方摘要 | 明确指出分头但共享输入特征仍有任务冲突，使用跨层特化与对齐。**直接创新性威胁**；不能以“分头还不够、分状态”一句话声称大gap。 |
| [BIG，CVPR2022](https://arxiv.org/pdf/2112.04222) | 复用091 F记录，本轮未重新通读 | 分类后多实例grounding，关系领域已有职责分工。本草稿的定位状态与预测支持语义读取顺序不同，但换顺序本身不保证创新。 |

没有声称检索了所有论文，没有跨任务拼接mAP。上述文献没有替本项目证明“分类梯度造成334残余”。近邻有效也不表示在本群体直接可复现。

## 5. 能形成什么贡献，尚缺什么

若E099通过，可能贡献是：在实体输入、候选容量与前向一致时，找到短事件定位受到共享分类更新干扰的可干预证据，并据此设计**事件状态先建立、语义定向读取**的关系预测框架。

目前不能写成贡献的内容：PCGrad、双头、任务解耦、mask pooling、stop-gradient、事件query本身；它们都有强先例。架构的新颖性仍为`HIGH_RISK / NOVELTY_UNCERTAIN`，只有新增小分支或套用梯度手术不足以构成CVPR方法创新。

## 6. 交付自检

- [x] E091/E097/E098负结果和未开训边界写入；没有改写为已证完整根因。
- [x] 只保留H099一条原因；只引用E099一个决定性合同。
- [x] 广泛历史调研＋本轮直接近邻核查；标注阅读层级。
- [x] 三文件已经写出；本轮不运行模型、梯度测量、容量审计或训练。
- [ ] 实验验证完整原因：**未完成**。
- [ ] 新架构效果与创新性：**未验证，未采纳**。

写完本轮三文件后停止，不自动排队100。
