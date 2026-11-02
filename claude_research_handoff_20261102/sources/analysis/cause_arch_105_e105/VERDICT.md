# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**（eta≪0.20；T−N 未全>0；且 T 与反转对照 X 命中完全相同）

实验：E105-SOFT-MASS-DECODE。先完成 residual synthesis + literature map，再开唯一假设。合同 SHA256 `14c83eb5e1248158abcd41e62bb33ecb08e4136836cd2a707c999941478f630b`。

## 设计边界

- 训练：仅 3 条原生轨迹（seed 100/101/102 × 500），**共享权重**；N/T/X 为解码臂。
- 准入：C0 hull 334=29 / soft=27 / revsoft=27；hull≠soft → 操纵可区分。

## 334 结果

| seed | T soft | N hull | X revsoft | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 33 | 35 | 33 | −2 | 0 |
| 101 | 41 | 36 | 41 | +5 | 0 |
| 102 | 41 | 37 | 41 | +4 | 0 |

- eta(T,N)≈**0.0079**；eta(T,X)=**0**；eta(T,C0-hull)≈0.032。
- C0 上 soft 略差于 hull（27 vs 29；主组 40 vs 46）。

**REJECT。** 不采纳 H105/M105。不重开 H100–H104。

范围：否决「在原生训练下，仅用 [0.1,0.9] 软质量分位替换 hull」作为本预算主因；T≡X 命中进一步削弱特异性。根因仍 OPEN。
