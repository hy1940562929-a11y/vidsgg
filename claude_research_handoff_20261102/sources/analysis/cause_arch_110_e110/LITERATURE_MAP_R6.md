# Literature map R6 — after H109

## O4 — Matching cost / assignment bias (selected)

| Work | Source | Relevance |
|---|---|---|
| DETR bipartite matching | https://arxiv.org/html/2005.12872 | Cost-driven GT↔query binding |
| RTD-Net relaxed temporal matching | https://arxiv.org/pdf/2102.01894 | Temporal matching structure for actions |

**Project:** Scale Hungarian mask/dice (and optionally class) costs by GT duration so T prefers short GTs, X prefers long. Loss coefficients unchanged (≠ H101/H102).

## Deferred

Permanent extra eval queries; encoder unfreeze.
