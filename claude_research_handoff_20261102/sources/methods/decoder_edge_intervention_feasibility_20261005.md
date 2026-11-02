# Decoder读取连接干预：只读源码核验

2026-10-05。Explorer核查，Commander保存。不是022上下文输入替换，不是已完成的科学实验；无机制结论、无方法或训练准入。

在不替换入口X的条件下，可仅屏蔽decoder query→memory的attention连接。原backbone/neck输出、f、编码memory数值和位置可保留；被屏蔽源时间的信息仍可能经其他路径传播，不能称已移除语义上下文。

## 实际结构

_mask_vrd将feats[-1]传入predictor，FPN输出另作f。predictor归一化并投影最粗特征为memory。四层decoder都读取同一尺度/同一memory，未轮换FPN尺度。当前memory为B×256×(Tpad/8)，9query、8head，f为B×256×Tpad。

DecoderLayer.forward接受attn_mask并传到MaskedMHCA_QKV；上层Decoder.forward和DecoderOnly.forward未透传，因此原正常路径为None。可在model.predictor.transformer.decoder.layers[i].multihead_attn四个实例的forward pre-hook/wrapper提供_attn_mask；不应全局替换该类，backbone也有相关attention。

attention logits为B×8×9×M，显式bool mask是B×9×M，True允许、广播所有head。显式mask分支替代原padding屏蔽，必须与original_memory_valid.expand(B,9,M)取AND。保持_kv_mask/memory_mask原样，避免改变K/V数值；每query保留至少一个有效token，否则全负无穷softmax产生NaN。

当前没有预测mask门控。全部decoder完成后才算class/mask；deep_supervision保留四层hs供辅助输出，不反馈给后层。只屏蔽后层不能清除早期query残差。

## 可解释性边界

- K/V先经kernel3、stride1时间卷积；允许邻位仍混合被禁位的memory。
- query层顺序self-attention→cross-attention→FFN；单query限制可能被其他query在后层重新带入信息。
- backbone已经上下文化，粗尺度列不是纯局部时间证据，没有现成唯一的信息边界。
- 末端e^T f不是attention softmax。固定f时，连接变化经hs→e改变mask，class也可能变。只能称该读取操作的受控干预，不能把代数分解当独立语义因果。
- 022保存X/M/e/f，未保存memory/K/V。因此未来若采用此路线，仍须新增memory/K/V/f不变与显式全允许mask无操作一致性检查；022不能替代这些检查。

## 源码定位

均相对analysis/vrdone_architecture_010/training_preflight/source_snapshot/models/：
- maskvrd.py:161，_mask_vrd输入分支。
- predictor.py:85/92/102，memory、decoder与末端读出。
- local_transformer.py:117–178，K/V卷积、显式mask及softmax。
- local_transformer.py:732/773/807，具体attention类、layer接口、自注意。
- local_transformer.py:875/956，上层接口。
- analysis/context_entry_022/torch_capture.py:72/93，现有导出范围。

后续仅待设计审查：等预算连接删除与匹配随机删除能否回答一个明确的小命题，何种结果应停止。没有据此启动新实验或正式选方向。

## Innovation最小设计评议（UNVERIFIED，未冻结）

可作为优先于缺乏可信供体的输入替换的M0候选，但相同删除数量并非充分公平对照。可证伪的小命题：对预先固定query及段外memory连接集合，关闭这些连接，相对不删除基线及匹配随机删除，改善正确目标固定读出，并产生超出仅段外hull变化的决策收益。它测试连接删除的总效应（含softmax重新归一化），不是消除全部段外信息。

主要混淆：随机删除可能破坏目标证据；定向删除优于受损对照不等于优于原模型。必须保护相同目标token集合，固定同层/同query删除预算，并记录原attention质量、距离、连续性。原attention质量不同会使同数连接的干预强度不同。若主臂删除全部段外连接，则保护目标的相同预算随机集合没有自由度，不能冻结这种比较。

正式科学协议前最低要求：全允许mask精确复现；X/memory/KV/f和有效padding不变；任何query不全屏蔽；token映射、query和删除集合事先固定；不按干预结果换query。报告相对基线的有方向改善、目标内mask/类别margin、原生输出全体得失，并有只更新段外mask的几何对照。

停止条件：只改变输出、只优于破坏目标的对照、收益由机械hull缩短解释，或总体损失抵消目标收益，不支持该小命题。有效匹配对照下没有可分辨差异，就停止“这些连接有特殊负作用”的解释，不增预算追结果。020/021不为该机制提供正证据。
