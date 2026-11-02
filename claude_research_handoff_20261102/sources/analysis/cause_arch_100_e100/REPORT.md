# E100 MEMBERSHIP_EXPAND_RERUN 报告（E100-A1）

日期：2026-10-21 06:06:19 EDT。监督端补写封存（Codex 训练与统计完成后因 usage_limit_exceeded 未能写根目录判决）。

## 1. 修订与成员

用户 2026-10-21 批准扩展 mechanism-dev。修订文本：[CONTRACT_AMENDMENT.md](attempt_04_membership_expand/CONTRACT_AMENDMENT.md)。

| | 修订前 | 修订后（冻结） |
|---|---:|---:|
| 视频 | 8 | 16（+8） |
| 事件 | 169 | 302（+133） |
| eligible | 61（短15/长46） | 115 |
| 共同支持 | 3 / 10 / 16 | **7 / 23 / 37** |

新增视频（哈希顺序第一块）：0006_2569620797, 1105_4410882051, 0047_2513208116, 1015_3880847745, 1130_7921064402, 1012_7186531887, 0042_7563661830, 1026_2927552798。

首块即达 §5.1，未追加第二块。评价成员与 H100 操纵不变。

## 2. 准入

- 重建：1745/1745，max abs=0（admission_v4）。
- 支持门：PASS（上表）。
- 浮点修复：保留。

## 3. 训练

N/T/X × seed100/101/102，各 500 步 → **9/9 完成，4500 优化器步**。详见 `attempt_04_membership_expand/imported_results/training_v1/`。

## 4. 判决对照表

来自 `final_statistics/verdict.json`：

```json
{
  "verdict": "REJECT",
  "subreason": "EFFECT_TOO_SMALL_OR_UNSTABLE",
  "checks": {
    "active": true,
    "headroom": true,
    "true_path_better": true,
    "eta": false,
    "seeds": false,
    "main400": false,
    "mechanism": false,
    "protection": false
  },
  "training_completed": 9,
  "optimizer_steps": 4500,
  "contract_amendment": "E100-A1"
}
```

334 上 T−N 净增：{'100': {'N': 6, 'X': 21}, '101': {'N': 0, 'X': 18}, '102': {'N': -2, 'X': 25}}

机制（末层相对对照的短组 AUROC 点/CI）摘要：vs N -0.0186544 CI[-0.07344219382909865, 0.030796155356729575]；短−长(T−N) -0.0277415 CI[-0.08423983006066271, 0.023581965789457433]。

## 5. 科学含义

在扩员后可识别的机制支持上，路径级联合分配（T）相对逐层独立分配（N）**未稳定改善** 334 残余短关系命中，亦未同时满足机制与保护交集。H100 本预算下 **REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE**。完整根因交付仍未成立；不自动开启下一 cause。

## 6. Codex 中断说明

turn `01a15dc9-84ef-72d3-9005-3143cf12fa70` 在写出根 STATUS/VERDICT 前报错：`usage_limit_exceeded`（提示 Oct 27, 2026 2:48 AM 后再试或购买额度）。统计与训练产物已在 AutoDL/本地 `attempt_04_membership_expand/`；本报告由监督端按冻结 JSON 补写，未改任何数字。
