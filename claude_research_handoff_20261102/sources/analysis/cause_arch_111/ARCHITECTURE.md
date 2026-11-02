# M111-SHORT-QUERY-BANK：永久 +3 短先验 query

```text
query_embed: expand 9→12 (copy ckpt 0..8; init 9..11)
N: slots 9..11 random (no span prior)
T: slots 9..11 += short_prior_vector (learnable, init toward short)
X: slots 9..11 += long_prior_vector (learnable, init toward long)
Eval: all 12 queries → hull candidates
```
