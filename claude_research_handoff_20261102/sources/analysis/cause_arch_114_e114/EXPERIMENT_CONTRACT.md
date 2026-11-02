# E114-TGIOU-MATCH

- Arms: N = native bipartite_match; T = native costs + λ_tgiou·(1−GIoU_1d); X = same λ with GT spans time-shuffled within clip.
- λ_tgiou = 2.0 (DETR/Moment-DETR default scale for GIoU term).
- Span from soft mask mass (center, 2·E[|t−cx|]); decode/eval = native mask>0.5 hull.
- Seeds 100–102 × 500 AdamW steps; predictor-only; E100-A1 support.
- Gates: eta(T,N)≥0.20; all seeds T−N>0 and T−X>0; else REJECT.
- Admission: rebuild≤4e-6; C0 46/29; match disagreement T vs N > 0 (or no multi-GT batches).
