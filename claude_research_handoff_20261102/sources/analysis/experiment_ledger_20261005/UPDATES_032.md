# 032补充记录

实际2000步×2，32训练视频/563GT、8开发视频/169GT；两臂同输入/初始化/顺序，仅加可见覆盖监督。

相对原local：平均可见tIoU0.598849→0.635066，110→112/169，新增9/169、损失7/169；unknown50均值0.864003→0.912951，49/50通过不变。仍低于anchor121/169与pooled120/169，判定LOCAL_REPAIR_ONLY_NO_EXPANSION。C恢复、官方mAP、短时长分组未评，不得填值。

完整报告：`analysis/visible_coverage_032/REPORT.md`；证据EV-VIDOR-VISIBLE-COVERAGE-032；独立审查PASS_WITH_CONCERNS。
