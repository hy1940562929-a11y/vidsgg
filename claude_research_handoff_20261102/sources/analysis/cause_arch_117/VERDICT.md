# REJECT

**子因：DIRECTION_WRONG_OR_HARMFUL**（eta≈−0.14；T residual334 ≈ 0/0/1）

实验：E117-CENTERNESS-MASK。合同 SHA256 `26130c76f308753d55a639e3ce930d57c53084475ffcf33ec2e3494cb06645c1`。

## 334

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 39 | 12 | **−39** | −12 |
| 101 | 0 | 40 | 20 | **−40** | −20 |
| 102 | 1 | 42 | 23 | **−41** | −22 |

eta(T,N)≈**−0.138**；eta(T,X)≈**−0.058**。

**关闭族：centerness / Gaussian `loss_mask` 重加权 focal+dice（H117）。** 中心聚焦监督在本协议下有害，不是短关系主因杠杆。
