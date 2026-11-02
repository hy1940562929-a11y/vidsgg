# Figure 2 — empirical motivation, v4

## English caption

**A brief relation event within a persistent entity pair.** The frames show subject #3 and object #4 before, within and after the highlighted *watch* annotation in VidOR validation video `1018_4199369046`. **(a)** Their GT tracks remain co-present for 57.29 s. The pair has a persistent *next_to* annotation and eight separate *watch* annotations. VrdONE's highest-scoring saved candidate for the mapped pair and *watch* spans 7.34–51.02 s. **(b)** Enlarging the dashed window reveals a 1.13 s target event inside that 43.68 s prediction. Its temporal IoU with the **full, unclipped** prediction is 0.026. This example motivates precise localization of transient occurrences within long-lived entity pairs while preserving persistent relations.

Gray/blue bars denote GT; blue identifies the target. Orange denotes the unchanged native-input prediction (video-wide rank 5). Slashes mark intervals that extend beyond the local window. Frame boxes identify GT entities for visualization; the model prediction uses native proposals. The complete-video panel includes all eight *watch* events, so the long prediction is not represented as an extension of only one annotated occurrence. The figure does not establish that the persistent relation causes the localization error.

## 中文图注

**持续共存实体中的短暂关系。** 三帧分别位于 VidOR 验证视频 `1018_4199369046` 中目标 `watch` 标注的前、内、后。**(a)** 主体 #3 与客体 #4 的 GT 轨迹持续共存 57.29 秒，具有持续的 `next_to` 标注和 8 段分离的 `watch` 标注。VrdONE 保存结果中，同一映射实体对、同一谓词的最高分候选覆盖 7.34–51.02 秒。**(b)** 放大虚线框区域后，可见目标事件仅持续 1.13 秒，却被包含在 43.68 秒的预测中；目标与**完整、未截断的预测区间**的 tIoU 为 0.026。该样例提出了准确定位短暂关系、同时保留持续关系完整性的问题。

灰/蓝条为 GT，蓝色突出目标，橙条为原生输入下保存的实际预测（视频内 rank 5）。斜线表示区间超出局部窗口。图中实体框是可视化身份标识，不表示给模型输入 GT 轨迹。全视频视图保留全部 8 段 `watch`，避免把多次事件画成一次事件的延伸。

## Evidence unchanged

- `case_manifest.json`, `case_gt.csv`, `case_predictions.csv` are byte-identical copies of the audited v2 materials.
- All three frame SHA256 checks pass. No generated video frames, inferred masks, synthetic scores, or desired-result timelines are drawn.
- GT `[1440,1474)` = 34 frames; prediction `[220,1529)` = 1309 frames; FPS = 29.97002997002997.
- Exact duration ratio = 1309/34 = 38.5. Exact tIoU = 34/1309 = 0.0259740259740.
- The local view covers [47.3,50.1] seconds; its clipped bar is a visual enlargement only and is not used to recompute tIoU.
- Full source/proposal-binding/candidate audit: [v2 evidence](../figure2_real_vidor_v2/CAPTION_AND_SOURCES.md). Original source hashes and frame receipts are preserved in the copied manifest.

## Design research and changes

This revision addresses a template-like presentation style, not a claim that visual styling can prove human authorship.

| Source actually consulted | Design observation / guidance | Applied here |
|---|---|---|
| [Nature Research Figure Guide: specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) — text read | Favor legible labels, explicit axes/units and restrained graphics; omit decoration, shadows and colored text. | Black labels, simple axes, no cards, no background bands, no headline statistics. |
| [Nature formatting guide](https://www.nature.com/nature/for-authors/formatting-guide) — text read | Keep figures simple and move explanatory material into legends where possible. | Remove repeated claims and the research-question banner; keep rationale in the caption. |
| [TriDet, CVPR 2023, Fig. 1](https://arxiv.org/pdf/2303.07347) — first PDF page downloaded, rendered and visually inspected | Compact marks, direct labels and a caption carrying the fuller explanation. | Use direct interval labels and one local enlargement. No TriDet artwork or method diagram is reproduced. |
| [VSGN, ICCV 2021, Fig. 1](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Video_Self-Stitching_Graph_Network_for_Temporal_Action_Localization_ICCV_2021_paper.pdf) — article text and figure caption read; screenshot unavailable | The motivation figure presents the relevant empirical comparison directly. | Let the native prediction/GT discrepancy carry the motivation. No VSGN data or figure is copied. |

The specific layout choice is editorial judgment. There is no universal test for an “AI-looking” figure. The prior three-column diagram, generic ideal-output panel, large KPI-like numbers and slogan footer were removed. The new figure is a single case-based evidence display with two explicit temporal scales.

## Deliverables

- `figure2_evidence_en.pdf` / `.svg` / `.png`: English paper version.
- `figure2_evidence_zh.pdf` / `.svg` / `.png`: Chinese reading version.
- English labels use Times New Roman; Chinese labels use Songti SC. Fonts are embedded in the 7-inch-wide vector PDFs. SVG text remains editable.
- `preview_en.png`, `preview_zh.png`: screen previews.
- `pdf_preview_en.png`, `pdf_preview_zh.png`: independent final-PDF renders used for visual QA.
- `validation.json`: source checks, export checks and final QA.

Rebuild with the existing bundled runtime:

```bash
/Users/xk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 docs/figures/figure2_evidence_v4/draw_figure.py
```
