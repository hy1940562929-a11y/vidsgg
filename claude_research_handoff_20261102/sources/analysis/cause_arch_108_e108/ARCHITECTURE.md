# M108-DN-SHORT：训练期短偏置去噪 query（并行 DN forward）

```text
Content path (all arms): native predictor(x,f) → 9 queries → Hungarian loss
DN path (T/X only, parallel second forward):
  DNEncoder(class, noisy/shuffled span) → K DN queries → 1:1 GT loss
Eval: content path only (identical native 9 queries)
X: same DNEncoder + slot count; GT time spans shuffled (or random if G=1)
```

并行 DN，避免与 content 自注意力串扰；content 初始化与 N 逐比特一致。
