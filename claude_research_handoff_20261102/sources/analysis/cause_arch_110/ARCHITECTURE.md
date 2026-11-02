# M110-DUR-MATCH：时长偏置 Hungarian

```text
cost'[:, gt] = cost[:, gt] * w(duration_gt)
T: w = (d / d_ref)^α  with α>0 → short cheaper (smaller w)
X: w = (d_ref / d)^α → long cheaper
N: w = 1
Then linear_sum_assignment; native CE+mask+dice on matched pairs.
```
