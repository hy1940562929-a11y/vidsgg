# E098-JOINT-READING 最终判决

**REJECT**

子因：**UNDERREADING_NOT_ESTABLISHED**。

按[冻结合同](PROTOCOL.md)的训练前现象门槛，334残余中低读取（R<0.5）仅为 **47/334＝14.07%（47/334）**，23视频整簇bootstrap的95%CI为 **[9.95%,18.55%]**（原值[0.0994757166,0.1854641644]）。合同要求CI下界>0.50，实际未通过。CI为10,000次整视频重采样的分位数，不对应单一整数分子/分母。

连接与GPU正常；原生N重建 **1725/1725批max abs=0**，4e−6门槛、类别排序和mask二值检查通过。新9-slot容量门槛也通过：334组U9=331/334，新协议C0命中29/334，可达缺口302/334。本轮没有再次撞上E097旧F/原H的149/334容量墙。

**N/J训练0/6，参数更新0步。** J算子效果、500步训练、撤回/恢复、训练后AUROC和保护组结果均NOT_RUN。这不是服务器故障，也不是“J训练无效”的阴性结果。

停止的是H098所要求的“严重低读取系统性覆盖多数残余”前提及当前原因配方。不能扩大成“注意力分配没有问题”“短关系语义证据充分”或“完整根因已识别”；M098不采纳。

E091几何主配方未复活，E097未原样重跑；未改成员、阈值、query数、容差或RQ。独立结果复核 **PASS**。不启动训练、不追加实验、不自动排队后续任务。

证据：[REPORT.md](REPORT.md)、[低读取汇总](results_v1/admission_v1/underreading_audit.json)、[逐GT分数](results_v1/admission_v1/per_gt_reading.csv)、[容量账](results_v1/admission_v1/capacity_audit.json)、[原生重建](results_v1/preflight_v1/status.json)、[材料身份](results_v1/admission_v1/material_identity.json)、[导入SHA](import_receipt.json)、[本地自检](selfcheck.json)。

写完本判决即停。
