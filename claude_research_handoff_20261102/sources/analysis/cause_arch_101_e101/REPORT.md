# E101-DURATION-EQUAL-FOCAL 报告

## 假设

H101：长共存网格上，均值归约的点式 mask focal 稀释短事件正支持；单位正质量 focal（T）应优于原生（N）与长偏置（X）。

## 与 H100 的差别

H100 改跨层 Hungarian 事件身份；本轮保持逐层独立匹配，只改 `loss_masks` 中 focal 的正点权重。E100 已 REJECT，本轮不重开。

## 执行摘要

见 [VERDICT.md](VERDICT.md)。远程产物：`/root/autodl-tmp/VrdONE-data/commander_cause_arch_101/run_v1/`（本地镜像本目录）。

关键数字：操纵短 T/N≈11.7；训练 9/9；334 上 T−N = −4/−4/−7。

## 解释边界

X（长偏置）在 334 上点估计高于 N/T，本身**不能**支持 H101，也不足以另立「应偏袒长事件」为新根因（未设为成功臂，无预注册机制链）。完整根因仍 OPEN（090：区间形成失败位置已定位，致因未识别）。
