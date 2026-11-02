# M109-COEXIST-FILM：共存长度 FiLM content queries

```text
coexist = N_valid/512  (or valid.sum/512)
  T: true length per sample
  X: permute lengths within batch
MLP → (γ, β)  with γ=1+Δγ, Δγ/β zero-init
q' = γ ⊙ q + β   on query_embed (C,9) broadcast per batch → (B,C,9)
Decoder(content q') → native loss+hull
Eval: same FiLM with true coexist length
```
