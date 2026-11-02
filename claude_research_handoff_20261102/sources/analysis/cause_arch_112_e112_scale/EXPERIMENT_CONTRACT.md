# E112-SCALE：H112 更大尺度重测合同

## Amendment vs E112
| Item | E112 | E112-SCALE |
|---|---|---|
| Seeds | 100–102 (3) | **100–104 (5)** |
| Steps/arm | 500 | **2000** |
| Optimizer updates | 4500 | **30000** |
| Sampling weights | as E112 | **unchanged** |
| Model/loss/hull | native | native |
| eta gate | ≥0.20 | ≥0.20 |
| Sign gates | all T−N>0 and all T−X>0 | same |

VERIFIED/SCOPED_SHORT_OVERSAMPLE_EFFECT 或 REJECT（若仍小或变号）。禁止改权重公式冒充尺度。
