# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**（eta≈0.011≪0.20）

**SCALE_FALSIFIED** — 小尺度 E114 全 seed T−N>0（eta≈0.034）在 5×2000 下**未复现**；出现负/零 TN，T−X 多为负。

实验：E114-SCALE-TGIOU-MATCH。合同 SHA256 `a4f62c08b696b7c1cd7cce8bafa3ef98e235224837869e65f1550234bc154229`。

## 334（5 seeds × 2000 steps）

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 38 | 34 | 42 | **+4** | −4 |
| 101 | 44 | 34 | 37 | **+10** | +7 |
| 102 | 31 | 39 | 33 | **−8** | −2 |
| 103 | 35 | 35 | 47 | **0** | −12 |
| 104 | 40 | 29 | 39 | **+11** | +1 |

eta(T,N)≈**0.011**；eta(T,X)≈**−0.007**（macro≈0.014 / 0.004）。

Admission：match_disagree T↔N≈0.016；rebuild=0。

**关闭族：temporal-GIoU-matching-only（H114 / H114-SCALE）。** 不重开小尺度 underpowered+ 作为已验证因果。匹配几何单独加 GIoU 不是主因果杠杆。
