# E117-CENTERNESS-MASK
N=native mask loss; T=Gaussian centerness-weighted focal+dice on GT span; X=same weights from shuffled GT (cx,w).
Match/CE/hull native. Seeds 100-102 × 500 AdamW. eta>=0.20; all T-N>0 and T-X>0.
