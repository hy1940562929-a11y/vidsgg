# M101-UNIT-POSITIVE-FOCAL：事件内单位正质量的 mask 监督

2026-10-21。**DESIGN_ONLY / UNVERIFIED / NOVELTY_UNCERTAIN / NOT_ADOPTED**。对应 [H101](CAUSE_STATEMENT.md)。

## 1. 设计对象

在冻结实体输入与 9-query decoder 的前提下，只改训练时 **已匹配事件** 的 focal 归约方式：

```text
原生：L_focal(e) = mean_{t∈valid} α_t (1−p_t)^γ CE_t
T：   令 P_e = {t: y_e(t)=1}∩valid，|P_e|=d_e
      对 t∈P_e 乘权重 w_e = (∑_{u∈valid} 1) / max(d_e,1)
      再对 valid 做加权平均（或等价：正点贡献总和与事件长度无关）
X：   对正点使用 w_e^X = d_e / mean_d_batch（长事件更大），总正质量与 T 对齐到同一批量表，但方向相反
推理：仍用末层 sigmoid>0.5 hull，不改容量
```

Dice、分类 CE、Hungarian 成本系数、层数、辅助匹配身份（原生逐层）均不变。

## 2. 文献对齐（覆盖 ≠ 已解决）

| 工作 | 相关点 | 与 M101 |
|---|---|---|
| RetinaNet Focal Loss (Lin et al., ICCV 2017) | 难例/类别不平衡；本代码已用 α,γ | 未处理**同一 mask 内短正支撑的时长稀释** |
| Mask2Former / MaskFormer | per-mask dice + focal，`/num_masks` | 事件间已等权；事件内 mean 仍偏长网格短事件 |
| VrdONE (2024) | 关系 mask 集合预测；本地即该损失 | 本设计检验其 focal 归约对短关系的副作用 |
| 类别/实例平衡分割 (e.g. weighted CE, OHEM) | 重加权正类 | 这里按 **事件时长** 归一正质量，不是类频 |
| AutoLoc / 边界对齐 / OT 时间定位 | 弱监督下内外对比与质量移动 | E091 已试几何辅助失败；本轮不叠加 W1 辅助，只改正点质量 |
| 090 单位质量草案 | 分布 + W1 + 矩读出 | 本轮刻意更小；不改读出 |

**不能写：** 首次发现 focal、首次做时长平衡、已解决 VidOR 短关系。若结果只是通用正样本重加权迁移且无短特异机制证据，novelty 有限。

## 3. 条件性贡献

若 E101 达 `VERIFIED`：在锁定长共存短关系群体上，**事件内单位正质量 focal** 是可操纵的短区间形成因素，且区别于长偏置对照与零更新。

VERIFIED ≠ 完整历史根因、≠ 正式采纳整网、≠ CVPR gap 已成立。REJECT 则停止本配方，不自动改读出复活 090。

## 4. 与已停配方边界

不复活 E091 几何辅助、E097 旧容量、E098 低读取、E099 梯度投影、E100 路径联合匹配。H097–H100 / M097–M100 不升格。
