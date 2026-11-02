# E117-CENTERNESS-MASK

- Arms: N = native focal+dice; T = native + Gaussian centerness `loss_mask` on GT span; X = same Gaussian from **shuffled** GT (cx,w) within clip.
- Gaussian: center = GT mask mass center; σ = max(1, 0.5·GT width_frames); weight renormalized so ∑w = ∑valid.
- Match / class loss / hull decode = native. Seeds 100–102 × 500 AdamW steps; predictor-only; E100-A1 support.
- Gates: eta(T,N)≥0.20; all seeds T−N>0 and T−X>0; else REJECT.
- Admission: rebuild≤4e-6; C0 46/29; T vs N mask-weight disagree on short GT > 0.
