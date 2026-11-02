# Literature map R5 — after H108

Closed clusters omit (matching, loss reweight, decode, x/f adapters, DN).

## O3 — Pair-level conditioning / FiLM (selected)

| Work | Source | Relevance |
|---|---|---|
| FiLM (Perez et al.) | https://arxiv.org/abs/1709.07871 | Feature-wise linear modulation by conditioning signal |
| Conditional DETR / query conditioning line | DETR variants | Queries as the natural place to inject task/context |

**Project fit:** Long-coexistence is the locked problem. Condition **content query embeddings** by pair support length (`N_valid`/512). Distinct from H106/H107 (tensor adapters) and H108 (DN). X = FiLM with **shuffled** coexist lengths in-batch.

## Deferred

- Duration-biased Hungarian costs (weak when G=1).  
- Encoder unfreeze (no cached backbone grads).  
- Permanent extra eval queries (capacity confound).
