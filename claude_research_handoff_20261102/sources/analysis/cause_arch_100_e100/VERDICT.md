# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**

实验：E100-DEPTH-EVENT-ASSIGNMENT / MEMBERSHIP_EXPAND_RERUN（修订 **E100-A1**）。

## 合同与成员

- 原合同 SHA256 `d9e3e010a08a7822f8f0d8a1e2f50c4ed9cccb6484def3340d118809126971bc` 保留不覆盖；有效合同 = 原合同 + [CONTRACT_AMENDMENT E100-A1](attempt_04_membership_expand/CONTRACT_AMENDMENT.md)。
- mechanism-dev：**8视频/169事件 → 16视频/302事件**（新增 8 视频 / 133 事件；名单见 [MEMBERSHIP_FREEZE](attempt_04_membership_expand/MEMBERSHIP_FREEZE.md)）。
- §5.1 共同支持：**7 视频 / 23 短 / 37 长**（门槛 4/20/20 **PASS**）。评价切片 400/334/163/1101/2573 未改。

## 技术与训练

- 原生重建 **1745/1745**，max abs **0**；容量与数值修复（N_NATIVE_LOSS / RENAME_COST）保留。
- 正式训练 **9/9**（N/T/X × seed100/101/102），各 500 步；优化器更新 **4500** 步。

## 科学结果（冻结统计）

334 残余上 T−N 每 seed 净增：seed100 **+6**，seed101 **0**，seed102 **−2**（未全部 >0）。

| 对照 | eta 点值 | 95% CI | 净增均值 |
|---|---:|---|---:|
| vs N | 0.00451977 | [-0.00913277, 0.015052] | 1.333 |
| vs X | 0.0677249 | [0.03597, 0.110545] | 21.33 |
| vs C0 | 0.0275938 | [-0.00546734, 0.055929] | 8.333 |

合同要求相对 N/X/C0 的 eta≥0.20 且 CI 下界>0、每 seed T−N 与 T−X 净增>0；**eta/seeds 未过**。主400时间/类别、§5.2 末层 AUROC/短特异性、长组与全体保护亦未同时通过（见 checks：eta/seeds/main400/mechanism/protection=false；active/headroom/true_path_better=true）。

**此为在 E100-A1 修订合同下对 H100 的科学 REJECT（效果过小或不稳定），不是技术失败，也不是新假设。** 不采纳 M100；不排队 H101。

证据：[summary](attempt_04_membership_expand/final_statistics/summary.json) · [verdict.json](attempt_04_membership_expand/final_statistics/verdict.json) · [REPORT](REPORT.md) · [execution_receipt](execution_receipt.json)
