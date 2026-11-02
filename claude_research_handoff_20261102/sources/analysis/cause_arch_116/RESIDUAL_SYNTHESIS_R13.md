# Residual synthesis R13 (after H115) — 2026-10-21

## Closed
H115 REJECT: ActionFormer TFPN-on-f; TN +2/+3/−7; close family.
H114 remains underpowered+ scale-eligible (eta≈0.034, all TN>0, TX weak).

## Lit gap / deconfound
H113 failed with **span decode + aux** together. Moment-DETR/DETR use continuous L1/GIoU **as training loss** while VrdONE can keep mask→hull at eval. Untested cleanly: **cxw L1+GIoU aux only**, hull decode unchanged.

## Select H116
Family: Moment-DETR span **auxiliary loss** (keep hull). X = shuffled GT cxw.
