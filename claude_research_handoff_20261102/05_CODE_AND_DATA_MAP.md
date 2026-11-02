# 05 代码、数据、接口与复用地图

## 随包代码

`code_reference/native_snapshot/`：从本地导入的030 source_snapshot复制的模型/loader/helper evaluator；不是本轮从作者最新仓库拉取。`code_reference/native_snapshot/configs/vidor.yaml`等来自另一个已核对的服务器快照，manifest记录各自来源，不假定完整上游仓库/版本统一。

重点：
- `models/maskvrd.py`：MaskVRD，forward_training、forward_test、inference、bipartite_match、loss。
- `models/predictor.py`：MaskedTransformerPredictor；query mask embedding与FPN特征einsum。
- `models/backbones.py`、`fpns.py`、`local_transformer.py`、`blocks.py`：BSA/时间主干与时间mask传播。
- `dataloaders/{vidor,vidvrd}.py`、`category.py`：类别顺序、配对、区间坐标与标签。
- `VidVRD_helper/evaluation/visual_relation_detection.py`与`utils/evaluate.py`：官方关系评价及包装。

`code_reference/cdse_kte_lgsad/`：relation_enhancement.py、vrdone_adapter.py、integration_example.py、prepare_text_anchors.py、config、29项测试与历史VALIDATION.json。它是完整可参考实现，但研究状态UNVERIFIED；历史CPU随机权重/合成数据通过不证明精度。

`sources/analysis/cause_arch_*/`：各实验实际run/loss/adapter/admission/decode脚本和报告。请先读判决再复用，避免继承已经失败的训练配方。

## 必须核对的张量与接口

- 时间特征常为[B,C,T]，query常为[B,C,Q]或[L,B,Q,C]，按原predictor实际检查；不凭图猜维度。
- 前景50类+背景；当前native背景index0，TopK用`pred_probs[...,1:]`，不要错位类别ID。
- mask logits是原始实数，focal要求logits；不要对sigmoid输出再做with_logits。
- `_mask`、valid/output_mask、loss_mask用途不同；GT时间网格只做同网格右padding，不能随意插值改变事件长度。
- `num_masks`等TorchScript接口要求Tensor，不能传Python int/float。loss按关键字调用，normalizer数值/设备/类型保持。
- 原生复现检查固定同输入/布局/dtype/TF32和padding；历史门槛maxabs≤4e−6，不按结果放宽。两次浮点loss不可严格Python `==`，记录abs/rel差并使用已约定技术容差。
- 秒单位=真实采样stride/FPS；512是最大token数，不是fps或秒数。半开帧区间、最后token+1与+stride不能悄悄混用。
- shared predicate evidence GT用同pair同p事件并集；query实例mask target用每个独立实例。周边负点排除该谓词全部合法事件、未知标签掩蔽。
- 不能把GT谓词/GT短长/GT最佳分量作为部署条件；oracle仅诊断。

## 现有数据切分与风险

早期train32：官方training，缓存350pair/563GT，仅26/50谓词有正例；adapter-dev8：89pair/169GT；eval GT27：2573GT，反复曝光。E100-A1扩mechanism-dev16视频/302事件，共同支持7视频/23短/37长，原400/334/163等评价成员未改。其他TICK训练180、Event207/213来自另一套协议，不能混为一套数据。

原始native_cache与checkpoint不随轻量包复制。应按 `CACHE_AND_RESOURCE_CHECKLIST.md`核对本机可用性，缺数据时提交可执行准备脚本及精确字段，不能用087 score NPZ伪训原生分支。

## 历史服务器路径（仅定位，不是授权/在线事实）

用户曾给出学校私人目录：
`/home/user/xiongke/cvpr_lab/tick/`
`/home/user/xiongke/cvpr_migrate/rebinding_project/VrdONE/`
`.../VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth`
`/home/user/xiongke/cvpr_lab/tick/{train,val}_gt_features/`
`tick/{evidence_repair,orca,scwd,ls_compete,short_rel,causal_pl_context}/`

这些路径本轮没有远程核实。历史AutoDL端口、socket/GPU编号均不作为当前入口，不提供密码，不自动连接。真正训练由用户在合法环境交接执行。
