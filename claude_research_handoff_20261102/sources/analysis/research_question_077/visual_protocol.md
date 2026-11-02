# 077 盲视觉核验素材

## 样本与呈现

输入为边界审计产生的固定 `blind_sample.csv`：40 条短关系且真实连续实体共同存在至少 10 秒的 target，40 条同谓词、关系时长差不超过 0.25 秒、共同存在不足 10 秒的 control。选样不读取模型预测或成败。若输入清单不同，以其 SHA256 及对应选样协议为准，不自动补样。

本脚本不再选择样本，只按固定种子 113 打乱已有样本的呈现顺序，生成 `audit_001` 等编号。图中只显示编号、视频、谓词、GT 起止帧和主体／客体身份；不显示 target/control、匹配组编号、模型输出、置信度或成败。红框为主体，蓝框为客体。框缺失表示该帧没有相应 GT 框，不能自动解释为视觉不可见。

每个样本取 7 帧：GT 开始前 0.5 秒、开始帧、内部四分之一点、中点、四分之三点、最后一帧、最后一帧后 0.5 秒。GT 采用 `[begin_fid, end_fid)`，内部最后帧为 `end_fid - 1`。越出视频部分固定裁到有效帧并记录；重复抽到同一帧不改采样规则。前后帧仅用于上下文观察，不能因为关系未标注就当作真正负帧。

每例一张 4 列 × 2 行 contact sheet，单格 320 × 224 像素，保留画面宽高比。脚本使用 OpenCV 按帧 seek，不解码整个视频；默认不生成视频片段。需要动态证据才能判断的关系（例如 watch、运动方向）应回看原视频，不能用静态 7 帧代替完整动作核验。

## 输出与审计记录

- `materials.json`：原视频和标注绝对路径、SHA256、请求帧与解码报告帧、图像 SHA256、GT 框有无、成功／失败。
- `materials.csv`：素材索引，不包含模型结果或 target/control。
- `blind_audit.csv`：视觉清晰性、遮挡、边界清晰性、标注可信度等字段全部留空。
- `summary.json`：输入清单及脚本 SHA256、成功数量、失败数量；生成素材不计为完成语义核验。

核验者只能在实际查看后填写结果。若由 AI 助手复核，`review_type` 写 `assistant_visual_review_nonexpert`，明确不是专家人工标注。无法判断填 unknown／needs full video，不能擅自填 yes/no。GT 框存在不是动作可辨识性的证据；GT 关系存在也不是标注一定正确的证据。

视频与标注的帧数、fps 或 GT 身份／时间不一致时该例失败并保留错误记录，不移动帧号或替换样本。任何失败均保留在素材完成率分母中。输出目录必须为空或不存在，避免覆盖已有核验材料。

## 调用

```bash
/root/miniconda3/envs/rebinding/bin/python visual_materials.py \
  --manifest /path/to/blind_sample.csv \
  --video-root /root/rebinding_project/VrdONE/datasets/vidor/videos \
  --annotation-root /root/rebinding_project/VrdONE/datasets/vidor/annotations/validation \
  --out /path/to/new/visual_materials
```

这只是材料制作，不运行模型、不训练，不生成任何语义标签。素材本身不能证明“短关系视觉证据不足”或“长共存压制短关系”。
