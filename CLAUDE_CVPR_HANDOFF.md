# Claude Handoff Pack — VidOR Short-Relation / VrdONE → CVPR 2027

**Handoff date:** 2026-11-02 (Asia/Shanghai, UTC+8)  
**Owner user:** LIEAST (t515u)  
**Purpose:** One-shot paste to Claude so Claude can (1) return concrete solutions + production-ready code, and (2) write a full CVPR-template paper, **assuming experimental numbers where needed** (clearly marked).  
**Source note:** Mac SSH `b405host1-xiongke` and school paths were **unreachable** at handoff build time (connection closed). Numbers below are from conversation-locked facts + bot memory/STATUS copies on the box. Prefer re-syncing school STATUS files when SSH returns; treat any conflict by preferring school STATUS_* over this pack only if school files are newer and consistent.

**School anchors (when reachable):**
- `~/xiongke/cvpr_lab/tick/short_rel/` — STATUS_*.md
- `~/xiongke/cvpr_lab/tick/dr_probe/` — STATUS.md, P0/P1b artifacts
- `~/xiongke/cvpr_migrate/rebinding_project/VrdONE/` — code
- Mac: `~/Documents/deepseek-harness/.../transient_vrd/cvpr2027_draft/` — DR_PROBE_NEXT_STEPS.md, SIGNIFICANCE_AND_FRAMING.md, MODULE_NOVELTY_REVIEW.md (not fetched)

---

## PASTE THIS FIRST (system-style prompt for Claude)

```
You are a senior CVPR researcher + production ML engineer. You have NO prior chat with me.
Everything after this block is the ONLY allowed factual base for numbers, paths, and decisions.

YOUR MISSION (one response or a short series of deliverables):
1) Implement on VrdONE a boundary-aware interval method: keep BSA path; optional TDiff OFF by default;
   NEW start/end boundary heads + duration prior / short-event loss weighting; decode from start–end
   (NOT first–last hull of a wide mask). Do NOT revive CDSE/KTE/LGSAD as the main method.
2) Deliver production-ready code: concrete file paths under the VrdONE tree, unified diffs / full new files,
   train + RelDet eval scripts, env flags, smoke checks.
3) Write a FULL CVPR 2027 paper in official CVPR LaTeX template style (sections complete).
   Use REAL measured numbers wherever listed as MEASURED in the handoff.
   Where experiments are not yet run, invent PLAUSIBLE tables/figures numbers but mark every such cell
   clearly as [ASSUMED] or [PLACEHOLDER] in captions/tables/footnotes. Never silently mix assumed with measured.
4) Framing: short-relation temporal inflation (wide plateau mask scores) → diagnosis that bottleneck is
   mask-head boundary precision / inside–outside contrast (not missing backbone boundary signal) →
   method = boundary-aware continuous interval parameterization + duration prior.
5) Include Related Work, method figure descriptions, ablation plan, limitations, ethics/broader impact.
6) End with an explicit experiment checklist the user must still run to replace [ASSUMED] with real RelDet.

HARD CONSTRAINTS:
- Final claims must prefer full official RelDet (VidOR val ~835), NOT val250-only diagnostics.
- val250 freeze-trunk numbers are diagnostic only; do not sell them as paper main-table SOTA.
- SoftMaxDur / SDRP / ORCA-full / PAEG / ER-replace / KTA-Event / CDSE+KTE+LGSAD-as-main = NEGATIVES / STOPPED.
- Keep TDiff optional; measured TDiff gain is negligible (~+0.03 mAP).
- Chinese narrative OK in planning comments; paper body in English; CVPR LaTeX English.

Output order preferred:
A) Architecture + loss + decode design (short)
B) Code patches (files + diffs + scripts)
C) Full paper LaTeX (CVPR style)
D) Experiment checklist + open risks
```

---

# Mission for Claude

一次性解决：

1. **方法落地**：在 VrdONE / MaskConvTransformerBackbone 上实现 **start/end 边界头 + duration prior + start–end 解码**；BSA 保留；TDiff 可选默认关；**不要**把 CDSE/KTE/LGSAD 当主方法复活。
2. **代码交付**：可直接合入学校仓库的补丁（新文件全文 + 关键文件 diff）、训练/RelDet 脚本、环境变量、冒烟步骤。
3. **论文交付**：完整 CVPR 2027 模板风格 LaTeX；**未测实验可假设数字**，但必须显著标注 `[ASSUMED]/`；已测数字必须用下方 MEASURED 表，不得改写。
4. **叙事**：短关系时间膨胀（宽平台）→ D/R 探针证明非纯 D、非“特征无信息”→ 瓶颈在 mask head 边界精度 → 边界感知区间参数化。

**CVPR 2027 日程（用户侧）：** paper registration ~Nov 10 AoE；full paper Nov 16；supp Nov 23。

---

# Hard constraints

| ID | Constraint |
|---|---|
| C1 | **禁止**仅用 val250 作最终论文主表声明；最终以 **full RelDet**（ckpt_vidor 协议）为准。 |
| C2 | Val proposal features：学校 **832/835** pkl；缺 3 个视频按 **empty preds** 公平对比。 |
| C3 | Half train GT features **DONE**：`GT_boxfeatures_training_half3500/` **3500/3500** pkls ~42G。 |
| C4 | **CDSE / KTE / LGSAD 三模块 STOPPED**（用户 t503u）；DeepSeek 判定新颖性不足。勿作主方法。包路径仅作历史参考：`/workspace/zt/cdse_kte_lgsad_20261029/`。 |
| C5 | TDiff 已测：相对 BSA-only **mAP +0.03（可忽略）** → 瓶颈不在输入相邻帧差分。 |
| C6 | SoftMaxDur / SDRP / ORCA-full train / PAEG / ER-replace / KTA-Event / HCC-as-main = **negatives / STOP**。 |
| C7 | KDR / GLF：**不要**按设计文档宣称短关系胜利；实测 freeze-trunk 下 α 变负、短关系平/差。 |
| C8 | Only-TPR（CC decode）短 +1.89pp 是 **diagnostic decode**，非训练方法主贡献。 |
| C9 | 用户确认失败模式：**主要是 (1) 原始时间分数过宽的 plateau**；**(2) sparse spike + first-last hull 桥接** 少见。 |
| C10 | 不启动本 handoff 内的训练；Claude 只交付代码与论文草案；真实训练由用户在学校 GPU 跑。 |
| C11 | 假设数字必须标注；已测数字不得“四舍五入美化”偏离下表。 |

---

# Problem statement

## Task

**VidOR video visual relation detection** with **VrdONE**（MaskVRD-style：轨迹特征 → BSA subject–object interaction → query class + temporal mask → interval decode）。

## Phenomenon

短时关系（约 **<3s**）在实体对长期共存时严重漏检 / 被预测成过长区间（**duration inflation / temporal localization error**）。

## Official RelDet baseline（MEASURED）

| Source | mAP | R@50 | R@100 |
|---|---:|---:|---:|
| Ours `ckpt_vidor` full RelDet (school) | **11.74** | **11.07** | **14.25** |
| Paper reported | ~11.86 | ~11.13 | ~14.21 |

Under same R@100 hits（STATUS_R_AT_K）：

| Slice | Share of GT | Slice recall / note |
|---|---:|---|
| Short `<3s` | **~38.55%** of GT | slice recall **~7.62%**（约占 hits 的 20.62%） |
| Long | — | R@100 **~18.41%** |

## User diagnosis confirmation

问：失败更像 (1) raw time scores 太宽，还是 (2) 稀疏尖峰后 first–last hull 把空隙填满？  
答：**主要是 (1) 宽平台（plateau）**；(2) 罕见。  
补充：关系未发生的帧仍高分 → **platform inflation**。

## Scientific positioning（论文类型）

- **诊断/机制 + 针对性方法**：不是纯 closed-set SOTA 刷榜，也不是 open-vocab 新范式。
- 关联：alone→C 短召回大落差成立；主机械因子 = **localization / duration inflation**；**不要**声称已证明语义 long-suppresses-short（H5 MANIPULATION_FAILED）。
- 早期诊断论文路径（ARR zero-train decode）仍可作对照；**当前主方法证据指向 start/end + duration prior**。

---

# Measured evidence (tables)

> Legend: **MEASURED** = real run; **DIAG** = freeze-trunk / val250 diagnostic (not paper RelDet main table); **ASSUMED** = Claude may invent later for draft only.

## M1. Official RelDet — baseline & TDiff（MEASURED, full RelDet）

Train protocol note (TDiff run): 180 vids / 1684 wins, 3368 steps, 2 epochs；insertion: after visual_embd→bbox fuse→mask, **before** BSA stem/mutual attn；shared TDiff；`VRD_TDIFF=1`；γ=0 → identity。

公式：`d_t = m_t m_{t-1}(f_t - f_{t-1})`, `d_0=0`；`h = ReLU(LN(Conv1D 1024→512, k=3))`；`add γ ⊙ h`。

| Run | Setting | mAP | R@50 | R@100 | ΔmAP vs B |
|---|---|---:|---:|---:|---:|
| R | Reference / reproduce family | 11.72 | 11.04 | 14.23 | — |
| B | BSA-only | **11.82** | **11.10** | **14.24** | 0 |
| T | BSA + TDiff | **11.85** | **11.14** | **14.28** | **+0.03** |

**读数：** TDiff 增益可忽略 → **瓶颈不在输入时序差分**。STATUS_TDIFF_PRE_BSA.md（学校；本 pack 未拉到原文）。

## M2. D/R Probe — P1b pixel / backbone ceiling（MEASURED, val-only 250 vids）

Goal: 区分 **D**（readout waste：边界已在特征里，解码没用）vs **R**（representation missing）。

| Probe | AUC / note |
|---|---|
| Pixel RGB-diff ceiling @ ±0.5s | **~0.65**；窗口加宽 → 更低 |
| Backbone pre/post-BSA | **~0.67**；short **~0.71** |
| MLP vs linear | MLP **≤** linear |

**读数：** Backbone ≈ pixel ceiling；**BSA 不稀释边界信号** → 不太像“表征里没有边界”。

## M3. D/R Probe — P0 mask logits failure modes（MEASURED）

| Stat | Overall | Short |
|---|---:|---:|
| PLATEAU failures | **~82%** | **~88%** |
| SPIKE failures | **~4.5%** | — |
| Median inflate (pred/GT dur) | **~2.7×** | **~10.5×** |
| Endpoints pulled outward | **~85%** | — |
| GT-inside / outside mask mean ratio | **≈1.0** | — |
| Hull → mass-run gain | **+0.5pp** | — |
| Oracle best-run gain | **~+2pp** | — |

**读数：**

- **不是主要 D**（hull 修几乎没用）。
- **不是**“特征缺信息”（对齐 ceiling）。
- **瓶颈 = mask head 学不会边界精度 / inside–outside 对比**。
- 与 **start/end 监督 + 连续区间参数化 + duration prior** 对齐。

## M4. Cause-Gate / coexistence association（MEASURED, frozen-candidate / diagnostic protocol）

| Item | Number |
|---|---|
| G0 alone→C gap | **~35.18pp**（433/1107 vs 285/7248） |
| H6 matched gap / β_C | **~33.88pp** / **~−33.60pp** → CONFOUND_NO |
| H1a localization oracle | G1 **~8.64pp**；closure **~0.754** → PASS |
| H1c OER alone→C | **~17% → ~71%**（Δ ~+54pp） |
| H5 semantic | **no claim**（MANIPULATION_FAILED） |
| Verdict | `CAUSE_OVEREXTENSION_PRIMARY` / `COMPETE_ASSOC_BUT_NOT_SEMANTIC` |

## M5. Features availability（MEASURED ops）

| Asset | Status |
|---|---|
| `vidor_per_video_val` proposal features | **832/835**；缺 3 → empty preds |
| `GT_boxfeatures_training_half3500/` | **3500/3500** ~42G **DONE** |
| Official RelDet now unblocked for eval with 832+empty protocol | Yes (fair compare) |

## M6. Prior diagnostic ablations（DIAG — freeze-trunk / val250; NOT paper RelDet）

### KDR（kinematic residual before visual_bbox_fuse）

- Long-train：`α_s≈−0.07`, `α_o≈−0.06`（变负，非期望 +0.1–0.3）
- Short time ≈ flat/worse（A→C Δshort time **−0.05pp**）

### Tri-Level（STATUS_TRI_LEVEL_ABLATION）

| Exp | short/long time (approx) | Note |
|---|---|---|
| Exp0 Baseline | 0.1120 / 0.5266 | — |
| Exp1 +KDR | 0.1062 / … | short **−0.58pp**；α more negative |
| Exp4 Only-TPR (CC decode, no retrain) | 0.1309 / long **−2.88pp** | short **+1.89pp** |
| Exp2 KDR+GLF | 0.1124 / 0.4795 | flat short / worse long |
| Exp3 Full KDR+GLF+TPR | 0.1327 / 0.4002 | long **−12.6pp** |

**结论：** 勿宣称 KDR/GLF 设计文档式短关系胜利。

### D0 / D0.1 / E1+ ladder

| Gate | Result |
|---|---|
| D0 κ=0 ARR-MBR | **time PASS** short **+1.61pp**（→~0.1278）；long time ~+0.18pp vs CC；用户笔记另含 **pred long −11.8pp** 类代价——decode 侧不完整 |
| D0.1 | **FAIL**（例如 protect/cc_pool 组合伤 long / short） |
| E1+ centre-hit | **STOP**：best centre-hit @±0.4L on S_C **12.3% < 25%** → 无 D1/D2 训练 ARR+E |

### PointRend try（STATUS_POINTREND_TRY）

- Zero-train mask_edge_off+cc topk=0.05：time short **+2.03pp**；**EXPLORATORY_WEAK_PASS**；无完整 StandardPointHead 训练。

### HCC / SoftMaxDur / ORCA / KTA

- HCC P0–P3 **STOP**（R median ~13≫6；预算不足）。
- SoftMaxDur/ADI/SDRP：**长杀 / 不足**。
- KTA short_traits S1 FAIL（kinematic peak real≈rand）。
- ORCA full path **STOPPED**；诊断 inflation 可保留。

---

# Failed / weak directions

Claude **不得**把下列当作主方法或主表 SOTA：

| Direction | Why dead / weak |
|---|---|
| CDSE + KTE + LGSAD | User STOPPED；novelty insufficient (DeepSeek) |
| TDiff-before-BSA as main fix | RelDet **+0.03 mAP** |
| KDR / GLF freeze-trunk | α→negative；short flat/worse |
| Full Tri-Level | long collapse |
| SoftMaxDur / SDRP / median-dur Event decode | long kill / pred-weak |
| ORCA-full / PAEG / ER-replace | STOPPED / FAIL |
| KTA-Event | S1 FAIL |
| HCC child search | P0/P1 FAIL |
| val250-only claims | User forbid for final |
| Semantic long-suppresses-short | H5 no claim |
| “Features lack boundary” / “BSA dilutes” | P1b contradicts |
| “Just fix hull / spike readout (pure D)” | P0 hull ~useless；spike rare |

**Still useful as paper negatives / diagnostics：** Cause-Gate, P0/P1b, TDiff null, Only-TPR/ARR decode upper bounds, PointRend exploratory.

---

# Recommended method (architecture)

**Name suggestion (Claude may refine):** Boundary-Aware Interval Parameterization (BAIP) / Start–End Relation Head on VrdONE.

## Keep

1. BSA path（subject–object mutual / stem attention）不变为主干。
2. Query **class** head（谓词/关系类）。
3. Optional **coarse mask**（可保留辅助，但不再是区间的唯一来源）。
4. Optional **TDiff**（`VRD_TDIFF`，默认 **0 / off**）。

## Replace / add

| # | Change |
|---|---|
| 1 | **NEW start head** `p_start(t)` 与 **end head** `p_end(t)`（每 query / 每对） |
| 2 | **Duration prior / short-event loss weighting**（按 GT 时长或谓词时长统计加权；短事件升权） |
| 3 | **Decode from start–end**：例如 `t_s = argmax/ soft expectation of start`，`t_e` 类似，或 joint score `score(i,j)=p_s(i)+p_e(j)+dur_prior(j-i)+cls`；**禁止**默认 first–last hull of sigmoid(mask)>τ |
| 4 | Supervision：GT start/end（可高斯/拉普拉斯软标签或 CE on boundary bins）+ 可选 mask BCE 作 auxiliary |
| 5 | 训练：优先半集 GT features `GT_boxfeatures_training_half3500` 起步，再扩全量；评估必须 **full RelDet** |

## Why this matches evidence

- Plateau + inside/outside ratio≈1 → 需要 **显式边界对比监督**，不是更强差分输入。
- Backbone≈ceiling → 不必先上重型表征模块（CDSE/KTE）。
- Hull/oracle-run 增益小 → 解码修不够，要改 **参数化与损失**。

## Method figure (describe for paper; Claude draws in LaTeX/TikZ)

Fig1 pipeline: RoI feats → (opt TDiff) → visual_bbox_fuse → BSA → {class, optional coarse mask, **start**, **end**} → **interval decode** → RelDet.  
Fig2 failure modes: plateau vs spike schematic with P0 %s.  
Fig3 D vs R probe cartoon: feature AUC vs mask ratio≈1.

---

# Codebase map

## Primary code (school)

```
~/xiongke/cvpr_migrate/rebinding_project/VrdONE/
  models/backbones.py          # MaskConvTransformerBackbone — wire heads here / nearby
  models/temporal_diff.py      # TDiff (exists after t505u)
  models/motion_residual.py    # KDR (historical; not main)
  eval.py                      # RelDet
  configs_vidor_reldet_school.yaml
```

## Features

```
vidor_per_video_val/                 # 832/835 proposal pkls
GT_boxfeatures_training_half3500/    # 3500 GT train pkls ~42G
```

## STATUS / probe labs

```
~/xiongke/cvpr_lab/tick/short_rel/
  STATUS_TDIFF_PRE_BSA.md
  STATUS_KINEMATIC_RESIDUAL.md
  STATUS_TRI_LEVEL_ABLATION.md
  STATUS_D0.md
  STATUS_D0.1.md
  STATUS_R_AT_K.md
  STATUS_MEGA_FEATURES.md
  STATUS_POINTREND_TRY.md
  (+ older STATUS_KTA, HCC, etc.)

~/xiongke/cvpr_lab/tick/dr_probe/
  STATUS.md
  STATUS_P1b* (if any)
  p0_summary.json
  p1b_results* (if any)
```

## Box copies (partial; for Claude offline reading)

- `/workspace/CVPR_DIAGNOSIS_HANDOFF.md` — older full diagnostic handoff (2026-10-27)
- `/workspace/CVPR_diagnostic_paper_writing_pack.md` — older writing pack
- `/workspace/short_rel/` — HCC/KTA STATUS & probes
- `/workspace/zt/cdse_kte_lgsad_20261029/` — STOPPED three-module package（勿主用）
- `/workspace/kinematic_residual/motion_residual.py` — KDR reference

---

# Implementation brief (interfaces, losses, decode)

Claude 应交付接近下列接口的代码（可改进命名，但语义保持）。

## 1. Config / env

```yaml
# configs snippet
model:
  use_tdiff: false          # VRD_TDIFF default 0
  use_boundary_heads: true
  use_coarse_mask: true     # auxiliary
  boundary_loss_weight: 1.0
  duration_prior_weight: 0.5
  short_duration_sec: 3.0
  short_loss_boost: 2.0     # example
decode:
  mode: start_end           # not mask_hull
  mask_hull_fallback: false
```

## 2. Heads（伪代码）

```python
# On BSA output e_so: [B, T, C] or per-query features
logits_cls = class_head(...)           # existing
logits_mask = mask_head(...)           # optional coarse [B,Q,T]
logits_start = start_head(e_so)        # [B,Q,T]
logits_end = end_head(e_so)            # [B,Q,T]
```

建议：`start_head`/`end_head` 为轻量 Conv1d/MLP+LN；与 mask 分头，避免共享最后一层导致再次平台化。

## 3. Losses

```text
L = λ_cls L_cls
  + λ_mask L_mask_aux          # optional Dice/BCE
  + λ_s L_start                # CE / focal on soft GT start
  + λ_e L_end
  + λ_dur L_duration_prior     # e.g. penalize | (e-s) - dur_pred | or match GT length distribution
  + short_boost * 1[dur_gt < 3s] * (L_start + L_end)
```

软标签：在 GT start/end 帧置峰，σ≈0.5–1.0s（按采样率换帧）。

## 4. Decode

```python
# Preferred
for each query:
  s = weighted_expectation(softmax(logits_start))  # or argmax + local refine
  e = weighted_expectation(softmax(logits_end))
  if e < s: swap or invalidate
  score = cls_prob * boundary_conf(s,e) * duration_prior_density(e-s)

# Forbidden as default
# interval = first_last(sigmoid(mask) > τ)   # classic hull
```

NMS / top-K 保持与 VrdONE RelDet 协议一致（通常 K≈9 queries/window——以仓库为准）。

## 5. Train / eval scripts（交付清单）

Claude 需给出：

1. `train_boundary_vidor.sh` — 半集 GT features、GPU、epochs、日志路径  
2. `eval_reldet_full.sh` — `eval.py` RelDet、832+empty 协议说明  
3. `smoke_boundary_forward.py` — 随机张量前向+反向  
4. Patch list：`backbones.py` / `maskvrd.py`（若损失在此）/ 新 `boundary_heads.py` / `decode_start_end.py`  
5. 明确 **不要**默认打开 TDiff；提供 `VRD_TDIFF=1` 消融开关即可  

## 6. Minimal acceptance smoke（用户跑）

- Forward shape OK；loss 有限；γ/开关不影响关闭模块。  
- 1 个 epoch 半集后 RelDet 不崩溃。  
- 主表对比：Baseline `ckpt_vidor` vs Boundary model（full RelDet）。

---

# Paper writing brief (CVPR structure, real vs assumed)

## Template

Official **CVPR 2027** LaTeX template（two-column）. English body. Claude outputs full `.tex` (and bib skeleton). Page budget：按当年 CVPR 主会页数（通常 8 pages excl. refs；以官网为准）；超页内容进 supp。

## Recommended title seeds

1. *Where Do Short Video Relations Fail? Boundary Readout, Not Missing Motion Features*  
2. *Platform Inflation in Video Relation Detection: From Diagnosis to Boundary-Aware Intervals*  
3. *Start–End Parameterization Beats Mask Hulls for Transient Visual Relations*  
4. （中文备忘，勿作投稿标题）短关系时间膨胀：从 D/R 探针到边界感知区间建模

## Abstract seeds（实测数字必须出现）

- VidOR + VrdONE RelDet mAP **11.74**（paper ~11.86）；short `<3s` 占 GT **38.55%** 但 slice recall **~7.62%**.  
- Mask failures：**~82% plateau**（short **~88%**）；median inflate **~2.7×**（short **~10.5×**）；inside/outside ≈1.  
- Feature boundary probe AUC **~0.65–0.71** ≈ pixel ceiling → not missing signal.  
- TDiff **+0.03 mAP** null.  
- Method: start/end heads + duration prior + start–end decode.  
- Main-table gains: **[ASSUMED]** until RelDet rerun.

## Section plan

| Sec | Content | Numbers |
|---|---|---|
| 1 Intro | Claim-first；成人–狗共存 vs 短暂 throw/feed 例 | MEASURED RelDet + short slice |
| 2 Related | VidVRD/VidOR/VrdONE；temporal localization & duration bias；mask/query decode；diagnostic papers | [CITATION NEEDED] OK |
| 3 Setup | VidOR，RelDet，short`<3s`，832/835 protocol | MEASURED |
| 4 Diagnosis | Cause-Gate；P0 plateau；P1b D vs R；user confirmation plateau | MEASURED |
| 5 Method | BAIP architecture；losses；decode；optional TDiff | equations |
| 6 Experiments | Protocol；checklist；tables | Mix MEASURED + [ASSUMED] |
| 7 Ablations | −start/end，−dur prior，hull vs start–end，TDiff on/off，mask-only | mostly [ASSUMED] |
| 8 Limitations / Ethics | GT-track vs detect-track；半集训练；假想安全与监控滥用一句 | honest |
| Supp | More P0 plots，val250 DIAG tables clearly labeled | DIAG |

## What MUST stay MEASURED in paper tables

- RelDet baseline 11.74 / 11.07 / 14.25  
- Paper 11.86 / 11.13 / 14.21  
- Short GT share 38.55%；slice recall 7.62%；long R@100 18.41%  
- TDiff B/T 11.82→11.85（+0.03）  
- P0 plateau/spike/inflate/ratio/hull gains  
- P1b AUCs  
- Cause-Gate G0/H1a/OER（可放诊断节或 supp；标明协议）  

## What Claude may [ASSUMED]

- Final mAP/R@K of BAIP vs baseline on full RelDet  
- Short-slice recall after method  
- Ablation Δs  
- Training curves，human qualitative figures captions  
- Runtime / params  

**规则：** 每个 [ASSUMED] 数字在表注写：*`Assumed for draft completeness; replace after RelDet run YYYY-MM-DD.`*

## Ablations plan（用户仍需实跑）

1. Baseline mask-hull decode（ckpt_vidor）  
2. +start/end heads，hull decode still（隔离监督）  
3. +start–end decode，no dur prior  
4. Full BAIP  
5. Full + TDiff  
6. Short-loss-boost on/off  
7. Freeze BSA vs finetune light  

## Limitations（必写）

- 半集 GT features 训练可能低估/高估全量。  
- 832/835 缺视频 empty-pred 协议。  
- 诊断多用 time-match；RelDet 是官方 detect-track 指标——二者不可偷换。  
- 未声称 open-world / OV-VRD。

## Ethics

VidOR 来自公开社交视频；方法可被用于监控场景——讨论 dual-use 一句；无额外隐私数据收集。

---

# Suggested title / abstract seeds

见上。额外中文 pitch（给用户，不进 LaTeX）：

> 我们证明 VrdONE 短关系失败主要是 mask 时间分数的宽平台膨胀，而非骨干缺少边界或简单 hull 读出浪费；相邻帧 TDiff 无效。因此用 start/end 边界头与时长先验替换宽 mask+hull，并在 CVPR 模板下给出完整诊断+方法稿；主表增益先按合理假设填写并标注，待全量 RelDet 替换。

---

# Open risks

| Risk | Mitigation |
|---|---|
| Start/end 头再次学成平台 | 强峰软标签 + short boost + 禁止 hull 捷径 |
| 伤 long relations | G4 风格门控：short↑ 且 long drop 有限；RelDet 分桶报 |
| 半集→全量特征域移 | 尽快扩全量 GT/proposal train features |
| 评审要求 SOTA | 诚实定位 diagnostic+mechanism；主表仍报 RelDet，不虚标 |
| 与历史 SCWD/ARR 叙事冲突 | 文中写 clear evolution：zero-train decode 有上限 → boundary parameterization |
| CDSE 包残留混淆 | 明确 STOPPED；代码默认不 import |
| 截止日期紧（Reg ~11/10 AoE） | Claude 先出完整草稿；实验 checklist 并行 |
| SSH/学校不可达 | 以本 pack 为准先写；联网后 sync STATUS_* |

---

# Explicit experiment checklist（replace [ASSUMED]）

用户 / Claude 指导用户必须跑完：

- [ ] Sync school STATUS_* + `dr_probe/p0_summary.json` 复核本 pack 数字  
- [ ] Smoke BAIP forward/backward on 1 batch  
- [ ] Train BAIP on `GT_boxfeatures_training_half3500`（记录 seed、steps、lr）  
- [ ] Full RelDet eval vs `ckpt_vidor`（832+empty protocol）→ fill main table  
- [ ] Short`<3s` / long slice recalls under same RelDet hits  
- [ ] Ablations 1–7 above（至少 4 与 5）  
- [ ] Optional：TDiff on/off confirm null on new code  
- [ ] Optional：re-run P0-style plateau rate on new model masks/boundaries  
- [ ] Register CVPR 2027 by ~Nov 10 AoE；full paper Nov 16；supp Nov 23  
- [ ] Replace every `[ASSUMED]` before camera-ready  

---

# Appendix A — Quick number cheat-sheet

```
RelDet ours:     mAP 11.74 | R50 11.07 | R100 14.25
RelDet paper:    11.86 | 11.13 | 14.21
Short GT share:  38.55% | short slice R: 7.62% | long R100: 18.41%
TDiff:           B 11.82/11.10/14.24 → T 11.85/11.14/14.28 (ΔmAP +0.03)
P1b:             pixel~0.65; backbone~0.67 (short~0.71); MLP≤linear
P0:              plateau~82% (short~88%); spike~4.5%; inflate~2.7× (short~10.5×);
                 end outward~85%; in/out~1.0; hull+0.5pp; oracle-run+2pp
Cause G0:        ~35.18pp; H1a closure~0.754; OER alone~17%→C~71%
TPR only:        short time +1.89pp (DIAG)
D0 ARR-MBR:      short time +1.61pp PASS; D0.1 FAIL; E1+ centre 12.3%<25% STOP
Features:        val 832/835; train half 3500/3500 DONE
STOPPED:         CDSE/KTE/LGSAD; SoftMaxDur/ORCA/KTA/HCC-as-main
```

---

# Appendix B — Sources unavailable at handoff build

At 2026-11-02 ~21:49 Asia/Shanghai：

- SSH `b405host1-xiongke` / school：**Connection closed**（失败）  
- Mac desktop agent：**offline**  
- Therefore **not freshly read**: school `STATUS_TDIFF_PRE_BSA.md`, `STATUS_KINEMATIC_RESIDUAL.md`, `STATUS_TRI_LEVEL_ABLATION.md`, `STATUS_D0.md`, `STATUS_D0.1.md`, `STATUS_R_AT_K.md`, `STATUS_MEGA_FEATURES.md`, `STATUS_POINTREND_TRY.md`, `dr_probe/*`, Mac `DR_PROBE_NEXT_STEPS.md`, `SIGNIFICANCE_AND_FRAMING.md`, `MODULE_NOVELTY_REVIEW.md`  

Numbers marked MEASURED come from user-locked conversation facts (t515u brief) + prior bot journal/memory written when those STATUS files were created. If school files differ slightly, **prefer school file + note delta**.

---

# Appendix C — Negative methods one-liners（论文 Related/Ablation 可用）

- SoftMaxDur/ADI: short 小升或平，long ≈−9–10pp。  
- SDRP: short ≪+3；OER 仍高。  
- KTA S1: kinematic peak hit real/rand ≈1.01。  
- TDiff: +0.03 mAP。  
- KDR: α negative。  
- Tri-Level full: long −12.6pp。  
- CDSE/KTE/LGSAD: novelty stop，未作主实验。  

---

**END OF HANDOFF — Claude: execute Mission; deliver code + CVPR LaTeX + checklist.**
