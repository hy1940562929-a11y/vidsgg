# E114-SCALE：H114 更大尺度重测合同

## Amendment vs E114
| Item | E114 | E114-SCALE |
|---|---|---|
| Seeds | 100–102 (3) | **100–104 (5)** |
| Steps/arm | 500 | **2000** |
| Optimizer updates | 4500 | **30000** |
| Match / λ_tgiou | native + λ(1−GIoU)；X=shuffled GT spans | **unchanged** |
| Decode/eval | mask→hull | hull |
| eta gate | ≥0.20 | ≥0.20 |
| Sign gates | all T−N>0 and all T−X>0 | same |

SCALE_SUPPORT 或 SCALE_FALSIFIED。禁止改 λ/匹配公式冒充尺度。
