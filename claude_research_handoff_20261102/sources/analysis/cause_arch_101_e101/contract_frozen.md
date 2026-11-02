# E101-DURATION-EQUAL-FOCAL：唯一决定性实验合同

2026-10-21。对应 [H101](CAUSE_STATEMENT.md)。**可执行**。Grace/supervisor 接管后冻结本文字；执行目录 `analysis/cause_arch_101_e101/`。

## 1. 唯一操纵与四臂

原模型 9 query、4 层；Hungarian 仍为**逐层独立**原生匹配（不采用 E100 的路径联合）。分类损失、Dice、cost_factor、背景权重、推理 hull 不变。唯一改变：`masked_sigmoid_focal_loss`（及 fuzzy 关闭路径下的同名非 fuzzy）在每个已匹配事件上的 **正时间点质量**。

定义：事件 e 的 valid 指示为 m_e(t)∈{0,1}，GT 二值 y_e(t)，正集 P_e={t: m_e(t)y_e(t)=1}，d_e=|P_e|，V_e=∑_t m_e(t)。点式 focal 记 φ_t（含 α,γ,CE）。

| 臂 | focal 归约 | 不变 |
|---|---|---|
| N | 原生：`(m_e ⊙ φ).sum() / V_e` 再对事件 `/num_masks`（即源码 `mean(1)`） | 原生对照 |
| T | 单位正质量：对 t∈P_e 令 w_t = V_e / max(d_e,1)；对负点 w_t=1；`∑(w_t m_e φ_t) / ∑(w_t m_e)`，再 `/num_masks`。d_e=0 时退回 N | 同匹配、Dice、分类 |
| X | 长偏置：对 t∈P_e 令 w_t^X = max(d_e,1) / V_ref；V_ref 取该 batch 内所有已匹配事件的 mean(V_e)；负点 w=1；同一归一化。使**更长**事件正点总贡献更大 | 同总表规模与可微路径，方向与 T 相反 |
| C0 | 预训练权重，零更新 | 共同参照 |

匹配成本矩阵仍用原生（未重加权）focal+dice+class，避免把“分配身份变化”混入本操纵。T/X 只进入 **loss_masks 的 focal 项**；Dice 保持 `masked_dice_loss` 原样。

**禁止：** 改 hull/阈值、加几何辅助、路径联合匹配、梯度投影、改 query 数、fuzzy 开关、选最佳 checkpoint。

## 2. 数据、成员、预算

- 训练/评价材料与 E100 同源：train32 / 原 cache；评价切片 **400 / 334 / 163 / 1101 / 2573** 冻结不变。
- events.json SHA256：`0c412de71d7ba10dbd1380798741c0069fb65557ee6659ff21e7992c2dc781fa`。
- cache_v1 manifest SHA256：`2613d524a528528212cb28c1d2a543431d7fc9b08cfc9d61f324a6aebd2f2269`。
- **机制-dev 成员**：直接继承 E100-A1 冻结的 **16 视频 / 302 事件**（见 `../cause_arch_100_e100/attempt_04_membership_expand/MEMBERSHIP_FREEZE.md`），以便 §5.1 有机会 ≥4/20/20；**不**再扩大。评价分母不随机制成员缩小。
- 三臂 × seed100/101/102，各 500 步；AdamW lr 3e−4、wd 0、clip 1；训练 decoder/query/norm/proj/分类头/mask embedding；冻 encoder/FPN/实体输入。
- AutoDL：`connect.bjb1.seetacloud.com:16241`。连接失败 → `TECHNICAL_FAILURE / REMOTE_CONNECTION`，不编造结果。

## 3. 执行前验收

1. 原生 logits 重建 train/dev/eval：全有限、max abs ≤4e−6；沿用 E100 的 float 容差与 RENAME_COST 确定性 fsum 修复，不得退回 exact `float==`。
2. C0 下 N forward 与仓库原生 loss 一致（容差内）；T/X 在 d_e≥1 的事件上 focal 数值与 N 可区分；对全正 mask（d_e=V_e）T 与 N 应在容差内一致（单位正质量退化为均值）。
3. 操纵暴露账（C0，全 train 批）：报告短（d_e 对应时长 <3s）与长事件的平均正点权重比 `mean(w_pos_T)/mean(w_pos_N)` 与 X 的对应比；若短组 T 的平均正点权重 ≤ N（在噪声容差内），走 `REJECT / INTERVENTION_INACTIVE`。
4. 容量：复用并核验 099/100 联合 9 槽见证（334 C0=29、U9=331 等）在**相同读出**下仍成立；不可核验 → 技术出口。
5. §5.1 共同支持：C0 固定末层原生 Hungarian 的 q0；负窗规则同 E100。需 ≥4 视频、短≥20、长≥20；不足 → `REJECT / MECHANISM_UNIDENTIFIABLE`，不静默改成员。

## 4. 推理与主终点

同 E100 §4：原生 9-query hull；时间形状 / 类别一致命中；eta(T,K)=mean_seed(M_T−M_K)/(331−mean_seed(M_K))，K∈{N,X,C0}；视频配对 bootstrap 10000、seed100。

VERIFIED 门槛（交集）：

- 334：eta(T,N)、eta(T,X)、eta(T,C0) 点值均 ≥0.20 且 CI 下界 >0；每 seed T−N 与 T−X 净命中均 >0。
- 400 时间与类别视频率差相对 N/X/C0 的 CI 下界均 >0；163 完整报告。
- 长 1101 两类视频率差 CI 下界 ≥ −1pp；全 2573 命中与类别 precision 相对 N/X/C0 不下降。

## 5. 机制测量

### 5.1 支持

同 E100：共同支持短/长；主比较仅共同支持。固定 q0 不进入推理选样。

### 5.2 分数改善

末层固定 q0：短组 AUROC(T)−AUROC(N/X/C0) 的 CI 下界均 >0；同视频同谓词短−长 (T−N) AUROC 差 CI 下界 >0；短组 m_4 点差相对 N/X/C0 均 >0。

另报：短组平均正 logit、正−邻域 margin；**不**把权重比本身当机制成功。

## 6. 互斥判决（顺序）

1. `TECHNICAL_FAILURE`（材料/4e−6/SSH/实现/预算）
2. `REJECT / SCOPE_NOT_TESTABLE` 或 `NO_RESIDUAL_HEADROOM`
3. `REJECT / INTERVENTION_INACTIVE` 或 `MECHANISM_UNIDENTIFIABLE`
4. `VERIFIED / SCOPED_UNIT_POSITIVE_FOCAL_EFFECT`：技术+操纵+支持+§4+§5.2 全过
5. 其余 `REJECT`：`TRUE_PATH_NOT_BETTER` / `EFFECT_TOO_SMALL_OR_UNSTABLE` / `METHOD_ONLY_NOT_CAUSE` / `PROTECTION_FAILED`

REJECT 停止本配方；不自动改读出并行第二假设。VERIFIED 仅限域监督效应，≠ 完整根因已识别。

## 7. 回执

封存合同 SHA、代码 SHA、重建误差、操纵暴露、支持账、9 个终点权重、预测、逐 GT 账、统计与唯一 VERDICT。本接管轮次在 `codex-feed-log.md` 记录 Grace/supervisor takeover。
