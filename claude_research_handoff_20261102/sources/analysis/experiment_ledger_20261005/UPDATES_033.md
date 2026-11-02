# 033固定checkpoint诊断

来源：`analysis/checkpoint_loss_033/REPORT.md`，EV-VIDOR-CHECKPOINT-LOSS-033，独立审查PASS_WITH_CONCERNS。

零训练/基座前向，32训练视频563GT、8开发169GT，同输入三个权重状态重放，dev精确复现032。训练已监督341组：coverage对anchor NLL5.804000→5.098893，IoU0.644770→0.621450，同GT反向118/341；unknown222组coverageNLL0.871977→0.209329，IoU0.996275→0.942472，同GT193/222。训练阈值458→449/563，+6/-15；dev121→112/169，+2/-11。

诊断正结果：退化不只在dev。方法收益不存在：032不扩展；不能唯一归因读出/损失，不能外推原VrdONE或C恢复。
