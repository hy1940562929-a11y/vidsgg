# E100 执行冻结：连接恢复续作

2026-10-21。旧连接失败记录保留在attempt_01_connection_failure。
本轮只执行原合同；N/T/X×seed100/101/102，500步；所有成员、准入、4e−6及出口不变。
原合同SHA256：`d9e3e010a08a7822f8f0d8a1e2f50c4ed9cccb6484def3340d118809126971bc`；逐字副本[contract_frozen.md](contract_frozen.md)。
先验收原生重建、容量、监督暴露及dev机制支持，任一门槛失败按合同停，不启动训练。

## 本轮停止回执

全模型原生重建1725/1725、max0及联合容量复核通过；准入脚本在train_0002.pt的重复原生loss严格相等检查停止。该额外零差值检查未记录数值差，不能作科学否定。未改合同或继续训练，结果见REPORT/VERDICT。

## ADMISSION_FIX_RERUN（2026-10-21）

用户授权仅修原生loss一致性检查。记录两次原生loss、绝对/相对差和独立直接criterion比较；绝对容差4e−6，rtol=0。若重复归约超出，固定同一输出与匹配，仅以Torch确定性算法重新计算loss，再按同一绝对容差验收并记录全部数值。不改forward、参数或科学门槛。禁止以无日志结果判PASS。旧脚本/报告归档于attempt_02_native_loss_check，旧results_v2不覆盖。

同类检查器数值修复：train_0014的X重命名前后选中32个成本项多重集合完全相同，分块float32求和差7.62939453125e−6；同项math.fsum重算差0。只把检查器成本累加换为确定性fsum，成本矩阵、Hungarian、loss、4e−6及科学门不变。原值、诊断和两版脚本保留。

## FIX_RERUN实际出口

损失111/111、重命名222/222、重建1725/1725 max0与容量复核通过。机制共同支持3视频10短16长未达4/20/20；按原合同REJECT/MECHANISM_UNIDENTIFIABLE停止，训练0/9。未放宽门槛、未自动扩样本。

## SUPPORT_EXPAND_RERUN（2026-10-21）

依用户指令，仅审计固定dev正负点与共同支持。独立集合实现与旧169/169事件一致；全dev只有15条eligible短关系，固定成员无法达到20短门槛。无合同修订、无成员或规则变更、无GPU，训练0/9。按固定dev不可扩停止条件结束；细目见REPORT及attempt_03_support_expand。

## E100-A1：用户批准 mechanism-dev 扩员

2026-10-21，原§2的8视频/169事件限制由[显式修订](attempt_04_membership_expand/CONTRACT_AMENDMENT.md)替代；其余科学门不变。按原哈希顺序追加8视频块；支持PASS前不训练。新增名单/事件/计数见本轮执行回执。

E100-A1最终机制名单冻结：首个8视频块已足够；新增133事件，累计16视频/302事件，eligible115、共同7视频/23短/37长，停止扩员。完整新增ID见attempt_04_membership_expand/expanded_membership.json。训练批次由random.Random(seed)对原88批逐轮打乱并截取500步，各臂使用同一列表和同seed原生随机状态。保留step500，不按dev选checkpoint。


## MEMBERSHIP_EXPAND_RERUN 完结（监督端封存 2026-10-21 06:06:19 EDT）

E100-A1 扩员后共同支持 7/23/37 PASS；准入 1745/1745；训练 9/9（4500 步）。科学出口 **REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE**（334 上 T−N 净增 +6/0/−2；eta(N) CI 跨 0）。Codex usage_limit 中断根目录写盘；监督端按 `attempt_04_membership_expand/final_statistics/` 补写 STATUS/VERDICT/REPORT/execution_receipt，未改统计数字。
