# M012：有界区间动作与相对效用

版本：M012-Bounded-Relative-Utility-v1。状态：UNVERIFIED，设计与实现阶段；尚无本轮训练或效果证据。Innovation提供只读设计，Commander整合并明确有效长度等实现语义。本轮不正式确定论文主方向、不采用为主方法。

## 已有证据与研究问题

M011的local开发端点NLL从6.350968降至6.167766，可见区间MAP tIoU却从0.701495降至0.598849。固定32视频中，其R100新增11、损失29，恢复原C 9条；有45,581个Top200前候选从原有效变为长度不足。pooled只取得总体净增1条的弱点估计。来源：`handoff/incoming/VIDOR-ARCHITECTURE-011/REPORT.md`、`reviews/vidor-architecture-011-results-audit.md`，证据`EV-VIDOR-ARCHITECTURE-20260929-011-PILOT32`。

研究问题：保留原预测并限制修改范围后，能否学会选择改善时间覆盖的修改，同时减少原正确关系损失？上述结果支持调查，尚未确定失败主要来自何种机制。

## 数学解释及可反驳诊断

### 分布拟合与决策损失不同

M011学习`P(s,e|x) ∝ exp(S_s+E_e) 1[s≤e]`并用MAP解码。NLL是分布拟合目标；MAP最优对应区间完全匹配的0–1损失。若目标为IoU，则理想的最小风险决策是`argmax_I E_{J~P}[IoU(I,J)]`，通常不等于MAP。NLL下降不能单独证明IoU提高，也不能推出NLL必然偏向短区间。

本轮先分组检查MAP的长度、对角概率质量、原/新跨度覆盖以及min5过滤路径。完整的同动作集MAP/MBR比较保留为后续诊断建议，不把本轮未实现的MBR诊断写成已完成结果。

### 先验约束与部署合法集

原跨度`I0=[s0,e0]`，`D=e0−s0+1`，`w=max(2,D/2)`。锚先验`U0(s,e)=−(|s−s0|+|e−e0|)/w`。任意内部单点相对原区间的先验惩罚都是`(D−1)/w<2`；无界残差可能使单点获胜，但这只是可能机制，须测量实际分数裕量。

训练/MAP允许`s=e`，原stride4解码器随后因`4(e−s)+1<5`删除它，是已核代码事实。零训练对照包括：原MAP、严格`s<e`的合法MAP、原MAP无效时回退原有效区间。无正mask或原无效仍保留统计分母；这些对照不是新模型训练成果。

### 保留约束的覆盖

M011显式KL仅覆盖未匹配query的Top6，未同样覆盖已匹配query的其他谓词及双端未知匹配实例；这些实例仍有锚先验。分组关联不足以证明因果。本轮以原动作显式保留、全部实际Top6候选的输出约束检验这一问题。

## 固定动作集

只修改原解码器有效候选，原无正点或长度不足保持原状。valid必须是连续prefix；不支持的输入立即报错，不静默改变原解码语义。

`r1=floor(D/8+0.5)`、`r2=floor(D/4+0.5)`，两端偏移分别取`{−r2,−r1,0,r1,r2}`。裁剪到有效prefix范围，再去重。原跨度作为动作0；其他按实际`(s,e)`升序。仅保留`Da=e−s+1≥ceil(D/2)`及`4(e−s)+1≥5`的区间。总动作不超过25；不使用银行家舍入。

这限制了修正范围，不能新增事件、轨迹、谓词或修复原无效候选，也可能无法到达M011的大幅修正正例。训练前action oracle量化容量，不在结果出来后扩大动作范围。

## 相对效用结构

复用冻结`F[B,256,T]`、`L[B,Q,T]`、`G[B,Q,51]`，Q9与Top6不变。`H=GELU(Linear256→64(LayerNorm256(F)))`。

原softmask `m=sigmoid(L)`和valid构成query前景池化u；先令`m_valid=m×valid`，再以`(maxpool9(m_valid)−m_valid)×valid`构成背景池化v，防止padding的logit经maxpool影响有效域。它们是代理特征，不是原decoder query hidden state。

动作表示`z_a=[u64,v64,mean(H_inside)64,mean(H_ring)64,embedding(p)16,geometry8]`，共280维。hard ring为动作区间两端各扩4点并去掉内部，裁剪到有效域。空环均值置0。

geometry8固定为：

1. `(s−s0)/D`；
2. `(e−e0)/D`；
3. `Da/D`；
4. `D/Tvalid`，Tvalid为有效prefix长度，不是padding长度；
5. 区间内m均值；
6. ring内m均值；
7. ring为空指示；
8. `(1[s=0]+1[e=Tvalid−1])/2`。

`h=MLP280→64→GELU→1`，`gain_hat(a)=tanh(h(z_a)−h(z_0))`。no-op增益恒0，末层零初始化，使最初严格保留原mask和最终输出。参数量35,809，需软件计数确认。末层bias在差值中抵消，不作为独立可识别参数。

仅当最佳非原动作`gain_hat>0.05`才修改，否则保持原预测。0.05是事先固定的工程门槛，不是误差界或已校准置信度，不扫描门槛。

## 监督与损失

沿用原G/L的固定Hungarian匹配。匹配实例真实谓词p、GT mask Y、已知域`K=valid & supervision_valid[p−1]`给出`J_K(I,Y)=|I∩Y∩K|/|(I∪Y)∩K|`，目标`gain(a)=J_K(Ia,Y)−J_K(I0,Y)`。

它是可见已知采样域的Jaccard覆盖，不等于正式min-vIoU。窗口外未知信息不作为负标签；双端未知仍可有可见覆盖监督。若`Y∩K`为空，跳过正损失且不降格为保留负例，单列计数。

P为原有效且固定匹配的真实谓词候选，包括真实p不在原Top6的情况，后者单列统计。R为全部原有效推理Top6候选去掉P。没有可见正监督的匹配p仍从R排除。

`Lutility = mean_c∈P mean_合法a≠0 (gain_hat−gain)^2`；`Lpreserve = mean_c∈R mean_合法a≠0 gain_hat^2`；`L=Lutility+1.0 Lpreserve`。分别先候选内再候选间平均。只有原动作的候选不进损失分母但保留统计；空集合返回可微零。正损失冻结为MSE，不使用曾讨论的SmoothL1。

保留项是向原动作收缩的正则，不是把所有其他Top6标成负GT。无正则且充分表达时平方损失对应条件平均效用差；本结构含正则，不能声称无偏或校准。

## 可保证与不可保证

在prefix、原过滤和输出接口核验通过后，构造保证原有效→无效及原无效→有效均为0，候选身份/数量、类别分数及排序预算不变，初始化最终输出等价。它不保证GT正确性、一对一匹配不变或mAP不下降。

若所有动作预测误差均≤ε≤0.05，则通过门槛的动作具有正真实代理增益；该误差前提没有实证保证，代理覆盖仍不同于正式检测目标。

## 容量、对照与结论边界

完整train/dev分别报告原有效、可编辑、Top6内外、单动作、最佳可达增益、正/零/负动作量、基线与oracle覆盖。任一split的Top6监督候选没有`best_gain>0.05`，停止训练，不追加动作范围搜索。该gate只证明潜在修正空间。

最多local/pooled两次训练。pooled仅以有效时间均值替代动作inside/ring视觉，空ring仍0；原前景/背景、mask、geometry都保留。这隔离动作局部视觉的增量作用，不是消除全部时序信息。

仍用固定已暴露32检测视频、1209GT、原C1047；C恢复且总体mAP/R100同时提高才过最低点估计门槛。只有同初始化、同样本顺序、实际优化步数相同，且local的mAP和R100均严格超过pooled，才得到当前架构/预算下动作局部视觉增量价值的探索性点估计支持；这不能证明局部视觉的必要性。若步数不等，只报告各自探索效果，不作该归因，不事后选步数或checkpoint。报告A/B损失、C/D恢复、全部分层和不确定性；不自动扩展到835或采用为主方法。

有限动作、质量预测和保留原解均有技术先例，NOVELTY_UNCERTAIN。相关数学背景可参考[GFL](https://arxiv.org/abs/2006.04388)的定位质量与训练/推理一致性及[TriDet](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html)的相对边界建模；这些论文不构成本项目效果证据。
