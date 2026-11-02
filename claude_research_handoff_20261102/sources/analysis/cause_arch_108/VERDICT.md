# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**（eta(T,N)=0；T−N 未全 seed >0；T−X 未全 >0）

实验：E108-DN-SHORT。R4 synthesis 后唯一假设。合同 SHA256 `672a485140c991d372affb93f94a12501ed7c9a78bcf64110b0320110ce72141`。

## 准入与训练

- Native rebuild maxabs=0；DN 辅助损失 T/X 均 >0；init DN mask L1(T↔X)≈0.084。
- 支持 E100-A1；C0 residual334=29、main400=46。
- 训练 **9/9**，4500 更新；推理均为原生 9 content query。

## 334

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 37 | 35 | 42 | **+2** | −5 |
| 101 | 36 | 36 | 33 | **0** | +3 |
| 102 | 35 | 37 | 32 | **−2** | +3 |

- eta(T,N)=**0**；eta(T,X)≈0.001；eta(T,C0)≈0.024。
- 并行 DN 监督未稳定抬高 content-query 短区间；打乱跨度对照 X 亦不劣于 T。

**REJECT。** 不采纳 H108。不重开 H100–H107。根因仍 OPEN。
