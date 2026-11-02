# 081：运行与口径

这是080之后的CPU探索，不是独立预注册，不训练、不重新推理。先冻结PROTOCOL.md和software_freeze.json，再运行8个连续终点对照与固定输出互斥分解。使用现有080结果CSV，所有原始数据保持只读。

```bash
cd /Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/week1_continuous_readout_081
/Users/xk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest test_run -v
/Users/xk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 run.py --out results_reproduction
python3 verify.py
```

run.py拒绝覆盖结果目录；verify.py独立核验已交付的results_v1点估计、成员、分解及SHA。可复现运行生成数值相同的表；最终报告另有scope说明补充，记录于report_scope_amendment.json，不改变统计数字。

**读出口径补充（核验077实现）**：固定selected是在同identity的候选中按目标谓词类别分数选出的query，不按当前GT时间挑选；本表只评它的时间形状。oracle是在该identity的全部连续段、全部query中，用GT时间取得最大tIoU，未要求该query正确预测目标谓词/class。因而oracle-only只表示“存在具有时间形状匹配的查询”，同时受GT时间oracle、类别未知、可能更换连续实体段三重限制；不能理解为选一个query即可恢复完整关系，更不是正式R@K。源码：`analysis/research_question_077/boundary_audit.py:316–327`。

- REPORT.md：简明实际结论。
- results_v1/summary.csv：8个连续终点配置，含支持量、加权分子分母、CI。
- results_v1/decomposition.csv：边际与三维互斥分解。
- results_v1/per_gt_partition.csv：逐GT标签。
- results_v1/H5_*.json.gz：完整共同支持分层和bootstrap结果。
- results_v1/manifest.json与根manifest.json：全部来源和文件哈希。
- verification.json：独立点估计和计数核验；未独立重抽bootstrap。

本轮所有H5配置的solo视频数不足20，无机制成立结论。oracle-only不是实际恢复数量。
