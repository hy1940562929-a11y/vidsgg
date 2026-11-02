# E097 容量门槛独立复核

Reviewer：context085_review。结论：**PASS，无阻断项；应按合同停止，不训练。**

只读核对容量脚本、冻结合同、源材料及输出；未重新执行穷举。

- 合法区间穷举和min5一致。
- 整数判据3×intersection≥predlen+GTlen等价于tIoU≥0.5。
- 保留原H准入；同slot选择同一区间的最大同时命中数，没有逐GT各选最优后相加。
- SHA、成员、slot绑定、保存的oracle区间和C0命中汇总一致。
- 334组C0=0，最大净恢复149/334，不超过167/334，触发REJECT/SCOPE_CAPACITY_CONFLICT。

限制：不能将该容量冲突写成局部状态对照机制已被否定。
