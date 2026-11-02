

## Numerical compatibility revision before training, 2026-10-14
First launch stopped before training; preserve results_v1. Both saved probes check all111 cached batches. All88 training batches remain bit-exact. Only adapter_dev batch22 with B=1 differs (maximum absolute3.814697265625e-6, mean3.018026717427347e-8). Actual sigmoid>0.5 masks, decoded hulls and four-layer ordered Hungarian assignments are unchanged. The new wrapper permits only this exact dev batch and absolute error<=4e-6 (no relative tolerance); all training and native inference gates remain strict. Old cached masks retain fixed matching authority. No scientific loss, seed, schedule, evaluation or continuation criterion changes. Output is results_v2. Independent reviewer admission: PASS_WITH_CONCERNS.
