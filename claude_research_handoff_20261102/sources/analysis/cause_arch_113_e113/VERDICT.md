# REJECT

**子因：DIRECTION_WRONG_OR_HARMFUL**（eta(T,N)≈−0.068；全 seed T−N<0）

实验：E113-MOMENT-DETR-SPAN。合同 SHA256 `fac157c0d082987eb29b32a4ccdbd209bb67e31f4d5c0bf1952d59cbbbad89e4`。

## 334

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 16 | 40 | 2 | **−24** | +14 |
| 101 | 22 | 40 | 2 | **−18** | +20 |
| 102 | 24 | 41 | 2 | **−17** | +22 |

eta(T,N)≈**−0.068**；eta(T,X)≈0.057。X≈floor → aux 相对 shuffle 有结构，但对 N/hull 有害。

Admission：ADMISSION_PASS（span_path_mask_L1≈0.080 记 exposure；训练全臂 `forward_with_span`）。

**关闭族：Moment-DETR-style cxw span 替换 hull 解码 + span L1 aux（H113）。** 非 UNDERPOWERED；不 scale。下一轮勿再以 span 解码替代 mask→hull。
