# 视频关系检测中“长共存下短关系”失效机理与 ORCA-Net 架构设计方案

## 1. 研究背景与核心实验现象

在视频视觉关系检测（Video Visual Relation Detection, VidVRD）任务中，模型需要从视频中检测出实体交互三元组 $\langle \text{Subject}, \text{Predicate}, \text{Object} \rangle$ 及其精确的起止时间区间 $[t_s, t_e]$。通过对代表性两阶段/图模型 **BIG (CVPR 2022)** 与单阶段端到端掩码模型 **VrdONE (ACM MM 2024)** 的代码复现与深入诊断，我们观察到以下关键实验事实：

1. **短关系的高占比与低召回困境**：
   时长较短（$\le 2.0\text{s}$）的瞬态关系（如 `touch`, `kick`, `pat`, `release` 等）约占所有关系三元组样本总量的 **30%**，是承载视频核心交互语义的关键部分，但现有模型在该子集上的召回率极低。
2. **长共存下短关系的断崖式崩塌**：
   在所有关系三元组中，约有 **10%** 的样本属于**“长共存实体中的短关系”**（即主客体在视频中长期同时在场 $|C| \gg 2\text{s}$，但瞬态交互仅发生在其中极短的片段 $|R| \le 2\text{s}$）。对比实验表明，模型在“独立短共存短关系（$S_{\mathrm{alone}}$）”上召回率可达 **39.1%**，而在“长共存短关系（$S_C$）”上召回率暴跌至 **3.9%**（差距高达 **$-35.18\text{ pp}$**）。
3. **首要机械死因——时间区间被拉长（Duration Inflation）**：
   诊断门控实验（Cause-Gate Analysis）证实，这种崩塌并非因为模型容量不足或长关系在语义上抹杀了短关系，而是因为**模型倾向于将短关系的预测时间区间沿着主客体的共存窗口大幅拉长**（过延展率 OER 从单独出现时的 **17.0%** 激增至长共存下的 **71.2%**，极端子集达 **94.5%**）。若使用定位 Oracle 修正时间区间，可消除 **75.4%** 的召回差距。

> **核心结论**：现有模型并非“认不出”短关系的谓词语义，而是深陷**“将短关系时间区间沿共存期拉长（Duration Inflation）”**的定位陷阱，导致预测区间与真值区间的交并比（$\text{tIoU}$）被严重稀释，无法通过 $\text{tIoU} \ge 0.5$ 的评价指标阈值。

---

## 2. 短关系检测难（区间被拉长）的四大根本原因

### 2.1 根本原因一：【表征层】“共存即关系”的捷径学习（The Co-presence Shortcut Illusion）
* **机理剖析**：
  在 VidVRD 数据集中，约 70% 的样本是中长时关系（如 `in_front_of`, `watch`, `hold`, `ride`），其真实关系区间与主客体轨迹共存区间（Co-presence Span）高度重合。无论是 BIG 的二部图分类-定位器，还是 VrdONE 的 Query 掩码解码器，在最小化全局训练损失时都会习得一条省力的**捷径（Shortcut）**：**只要检测到主客体同时出现且空间靠近，就在整个共存时间段内维持高置信度响应**。
* **对照印证**：
  * 在**短暂同框（$S_{\mathrm{alone}}$）**场景下，主客体仅共存 $2\text{s}$，短关系持续 $1.5\text{s}$。模型即便把整个 $2\text{s}$ 同框期都预测为关系，其 $\text{tIoU} = 1.5 / 2.0 = 0.75 \ge 0.5$，**掩盖了模型不会精确定位的缺陷**；
  * 在**长共存（$S_C$）**场景下，主客体同框 $20\text{s}$，短关系仅 $1\text{s}$。模型继续沿用“共存即关系”的捷径，导致预测区间膨胀至 $10\sim 20\text{s}$，定位缺陷彻底暴露。

### 2.2 根本原因二：【特征层】主客体“共存基底”的同质化污染与低通平滑（Feature Homogenization & Low-Pass Smoothing）
* **机理剖析**：
  BIG 与 VrdONE 的每帧输入特征均由**主体外观 $f_s(t)$ + 客体外观 $f_o(t)$ + 相对空间几何 $\theta_{so}(t)$** 构成。在一段 $20\text{s}$ 的长共存轨迹中，主客体的身份类别、外观纹理乃至相对距离在 95% 的时间内高度恒定（例如人一直站在狗旁，第 $10\text{s}$ 摸了一下狗 `touch`）。
* **平滑效应**：
  经过 VrdONE 的 SOS 跨帧注意力（Cross-Attention）或 BIG 的全轨迹图消息传递（Graph Message Passing）后，时序特征经历了多轮全局加权平均（等效于低通滤波）。真实发生交互的那 $1\text{s}$ 高频运动脉冲被周围 $19\text{s}$ 高度相似的静态共存特征同化，导致解码器在时间轴上看到的是一个**平坦的“高原响应（Plateau Response）”**，无法区分“真交互帧”与“仅共存背景帧”。

### 2.3 根本原因三：【解码结构层】VrdONE 的“掩码凸包陷阱”与 BIG 的“全轨迹回归失配”
* **VrdONE 的凸包陷阱（Mask-Hull Trap）**：
  VrdONE 预测一维逐帧掩码 $\hat{M}_t \in [0, 1]^T$，后处理采用固定阈值（$>0.4$）加**外接凸包（Outer Convex Hull）**：
  $$\hat{\mathcal{I}} = \big[\min\{t \mid \hat{M}_t > 0.4\},\; \max\{t \mid \hat{M}_t > 0.4\}\big]$$
  在 $20\text{s}$ 的长共存窗口中，由于主客体一直紧挨着，只要第 $2\text{s}$ 和第 $18\text{s}$ 各有哪怕一帧的响应分数略微超过 $0.4$，取凸包操作就会把中间整整 $16\text{s}$ 全部吞并，将 $1\text{s}$ 的短动作硬生生拉长成 $16\text{s}$。
* **BIG 的全轨迹归一化失配（Full-Tubelet Grounding Mismatch）**：
  BIG 在 Grounding 阶段直接对整条共存轨迹的归一化区间 $[0, 1]$ 做全局边界回归。当共存轨迹长达 600 帧而目标短关系仅有 15 帧时，真实区间仅占归一化全长的 $2.5\%$。由于缺乏局部锚点机制和谓词时长先验，全局回归器必然向全轨迹均值退化，输出宽大的模糊区间。

### 2.4 根本原因四：【优化与评价层】$\text{tIoU} \ge 0.5$ 的“非线性断崖”与尺度先验缺失
* **数学断崖效应**：
  设真实关系时长为 $L_{\mathrm{gt}}$，假设模型完全覆盖了真实区间，但在长共存窗口内向外多延展了 $\Delta L$（即预测总长 $|\hat{\mathcal{I}}| = L_{\mathrm{gt}} + \Delta L$）。要通过评价指标 $\text{tIoU} \ge 0.5$，必须满足：
  $$\text{tIoU} = \frac{L_{\mathrm{gt}}}{L_{\mathrm{gt}} + \Delta L} \ge 0.5 \iff \Delta L \le L_{\mathrm{gt}}$$
  * 对 $L_{\mathrm{gt}} = 15\text{s}$ 的长关系，向外多拉长 $10\text{s}$ 依然能达标（$\text{tIoU} = 15/25 = 0.6 \ge 0.5$）；
  * 对 $L_{\mathrm{gt}} = 0.8\text{s}$ 的短关系，**允许的最大拉长误差 $\Delta L$ 仅有 $0.8\text{s}$**！只要被拉长到 $2.0\text{s}$，$\text{tIoU} = 0.8/2.0 = 0.4 < 0.5$，立即被判为 **0 分（彻底漏检）**。
* **监督信号缺失**：
  传统的 BCE、Dice 和 L1/GIoU 损失既没有引入**“谓词固有物理尺度先验”**（例如模型把本该只有 $0.5\text{s}$ 的 `kick` 预测成 $15\text{s}$，未受到任何物理违和惩罚），也没有将“同一共存轨迹内的非交互帧”作为硬负样本进行对比抑制。

---

## 3. 针对性创新方法与模型架构设计：ORCA-Net

针对上述四大根本原因，同时吸取前期实验教训（**绝不增加额外 Query 容量，绝不拆分独立小样本分类头以免破坏谓词分类语义，绝不在测试集泄漏先验**），我们提出 **ORCA-Net（Orthogonal Coexistence-Debiased & Elastic-Anchored Network，正交去共存偏置与弹性锚定网络）**。

### 3.1 创新模块一：共存基底正交分解器（COD: Coexistence-Orthogonal Decomposer）
* **设计目标**：直接攻克**【根本原因一 & 二】**，在特征空间中将“主客体长期共存的静态基底”与“瞬态交互冲激”正交分离，消除高原响应。
* **输入与输出**：
  * **输入**：主客体融合特征序列 $\mathbf{E}_{so} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_T]^\top \in \mathbb{R}^{T \times D}$。
  * **输出**：平行稳态流 $\mathbf{E}^{\parallel} \in \mathbb{R}^{T \times D}$ 与正交瞬态流 $\widetilde{\mathbf{E}}^{\perp} \in \mathbb{R}^{T \times D}$。
* **内部数学计算**：
  1. **提取单位静态共存基底（Static Coexistence Basis）**：
     $$\bar{\mathbf{e}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{e}_t, \quad \mathbf{u}_{\mathrm{co}} = \frac{\bar{\mathbf{e}}}{\|\bar{\mathbf{e}}\|_2 + \epsilon} \in \mathbb{R}^D$$
  2. **正交子空间分解（Orthogonal Subspace Projection）**：
     $$\mathbf{e}_t^{\parallel} = \big(\mathbf{e}_t^\top \mathbf{u}_{\mathrm{co}}\big)\mathbf{u}_{\mathrm{co}}$$
     $$\mathbf{e}_t^{\perp} = \mathbf{e}_t - \mathbf{e}_t^{\parallel} = \big(\mathbf{I} - \mathbf{u}_{\mathrm{co}}\mathbf{u}_{\mathrm{co}}^\top\big)\mathbf{e}_t$$
  3. **局部运动能量增强**：
     $$\widetilde{\mathbf{e}}_t^{\perp} = \mathrm{LayerNorm}\Big(\mathbf{e}_t^{\perp} + \mathrm{MLP}(\Delta \mathbf{e}_t)\Big)$$

### 3.2 创新模块二：谓词条件化的锚点-高斯弹性门控头（PAEG: Predicate-Conditioned Anchor-Elastic Gate）
* **设计目标**：直接攻克**【根本原因三】**，在不改变原有 $K=9$ 个 Query 谓词分类能力的前提下，用自适应高斯包络自动截断长共存窗口远端的虚假掩码激活，根除凸包拉长。
* **内部数学计算**：
  1. **正交冲激锚点定位（Impulse Center Anchoring）**：
     $$\mathcal{A}_k(t) = \mathrm{Sigmoid}\left(\frac{\mathbf{q}_k^\top \mathbf{W}_a \widetilde{\mathbf{e}}_t^{\perp}}{\sqrt{D}}\right), \quad t_k^* = \sum_{t=1}^{T} t \cdot \frac{\exp(\beta \mathcal{A}_k(t))}{\sum_{\tau=1}^{T} \exp(\beta \mathcal{A}_k(\tau))}$$
  2. **谓词先验锚定的弹性宽度回归（Prior-Anchored Elastic Width Regression）**：
     $$\log \hat{d}_k^{\mathrm{prior}} = \sum_{c=1}^{C} \hat{p}_{k, c} \cdot \log d_c^{\mathrm{med}}$$
     $$\hat{w}_k = \exp\Big(\log \hat{d}_k^{\mathrm{prior}} + \tau_w \cdot \tanh\big(\mathrm{MLP}([\mathbf{q}_k \,\|\, \widetilde{\mathbf{e}}_{t_k^*}^{\perp}])\big)\Big)$$
  3. **高斯防膨胀软门控调制（Scale-Routed Gaussian Mask Gating）**：
     $$\mathcal{G}_k(t) = \exp\left(-\frac{(t - t_k^*)^2}{2(\alpha_w \hat{w}_k)^2}\right), \quad \hat{M}_k^{\mathrm{final}}(t) = \hat{M}_k^{\mathrm{raw}}(t) \cdot \big[\mathcal{G}_k(t)\big]^{\gamma_k}$$

### 3.3 创新模块三：共存内硬负样本对比与防膨胀校准（ICAC Loss & Calibration）
* **设计目标**：直接攻克**【根本原因四】**，在训练阶段强迫模型压低同一共存轨迹内非动作帧的响应，并在推理排序时惩罚异常膨胀区间。
* **内部数学计算**：
  1. **共存内硬负样本对比边际损失（Intra-Coexistence Contrastive Margin Loss, $\mathcal{L}_{\mathrm{icm}}$）**：
     $$\mathcal{L}_{\mathrm{icm}} = \frac{1}{|S_C|} \sum_{i \in S_C} \max\left(0,\; m - \frac{1}{|\mathcal{I}_{i,\mathrm{gt}}|}\sum_{t \in \mathcal{I}_{i,\mathrm{gt}}} \hat{M}_i^{\mathrm{final}}(t) + \frac{1}{\eta}\log \sum_{t' \in \mathcal{I}_{i,\mathrm{co}} \setminus \mathcal{I}_{i,\mathrm{gt}}} \exp\big(\eta \hat{M}_i^{\mathrm{final}}(t')\big)\right)$$
  2. **非对称时长膨胀惩罚损失（Asymmetric Duration-Inflation Loss, $\mathcal{L}_{\mathrm{infl}}$）**：
     $$\mathcal{L}_{\mathrm{infl}} = \frac{1}{N_{\mathrm{pos}}} \sum_{i=1}^{N_{\mathrm{pos}}} \max\left(0,\; \log \frac{|\hat{\mathcal{I}}_i| + \epsilon}{\max(|\mathcal{I}_{i,\mathrm{gt}}|,\; d_{p_i}^{\mathrm{med}})}\right)^2$$
  3. **防膨胀置信度重校准（Inflation-Calibrated Inference Scoring）**：
     $$\tilde{s}(c) = s_N(c) \cdot \exp\left(-\lambda \cdot \gamma_c \cdot \max\left(0,\; \log \frac{|\hat{\mathcal{I}}_c|}{d_{p_c}^{\mathrm{med}}}\right)\right)$$

---

## 4. 问题—病因—模块对位总结表

| 诊断维度 | 发现的具体现象与数字 | 根本机械原因 | ORCA-Net 对应模块 | 核心数学机制与贡献 |
| :--- | :--- | :--- | :--- | :--- |
| **共存对比反差** | 单独短关系召回 **39.1%** vs 长共存短关系 **3.9%**（差 $35.2\text{ pp}$） | 模型学习了“同框即关系”的共存边界捷径，未剥离静态同框 | **COD 正交分解器** | $\mathbf{e}_t^{\perp} = (\mathbf{I} - \mathbf{u}_{\mathrm{co}}\mathbf{u}_{\mathrm{co}}^\top)\mathbf{e}_t$，代数级清空恒定共存基底 |
| **时间区间膨胀** | 长共存下 OER 从 **17.0%** 飙升至 **71.2%**（极端达 **94.5%**） | VrdONE 外接凸包 $[min, max]$ 与 BIG 全局回归将共存期首尾噪声连成一片 | **PAEG 锚点-弹性门控** | $\hat{M}^{\mathrm{raw}}(t) \cdot \mathcal{G}(t; t^*, \hat{w})^{\gamma}$，用谓词先验宽度软剪裁远端膨胀尾巴 |
| **长短关系兼顾** | 独立短关系分类头易导致少样本谓词崩溃或挤占长关系配额 | 割裂分类头会丢失主干语义，增加 $K$ 会造成不公平候选膨胀 | **共享 Query + 软路由指数 $\gamma_k$** | 保持固定 $K=9$ 与共享分类头，$\gamma_k \in [0, 1]$ 实现长短平滑过渡 |
| **评价断崖效应** | $\text{tIoU} \ge 0.5$ 下定位 Oracle 可闭合 **75.4%** 的召回差距 | 短关系允许的绝对延展误差 $\Delta L \le L_{\mathrm{gt}}$ 极窄，传统损失不抑制同轨迹背景 | **ICAC 对比与防膨胀损失** | 显式打压 $\mathcal{I}_{\mathrm{co}} \setminus \mathcal{I}_{\mathrm{gt}}$ 内的硬负样本帧，严惩超时长膨胀 |
