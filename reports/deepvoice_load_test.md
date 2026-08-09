# Deepvoice API Load Test

- endpoint: `http://127.0.0.1:8000/api/v1/anti-spoofing/detect`
- audio: `daughter_register_01.m4a`
- audio bytes: 105586
- one unreported warm-up request was sent before measurement

- expected result signature: `(False, 0.2817, 5)`

| concurrency | requests | success | errors | result mismatch | throughput req/s | mean ms | p50 ms | p95 ms | max ms | model mean ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | 0 | 1.99 | 502.52 | 502.52 | 502.52 | 502.52 | 448.55 |
| 5 | 5 | 5 | 0 | 0 | 2.214 | 1599.29 | 1824.28 | 2258.32 | 2258.32 | 1483.43 |
| 10 | 10 | 10 | 0 | 0 | 2.531 | 2505.22 | 2502.44 | 3949.61 | 3949.61 | 2293.77 |
| 20 | 20 | 20 | 0 | 0 | 2.568 | 4473.94 | 4472.99 | 7768.23 | 7783.78 | 4135.04 |

This is a local-machine benchmark, not a production capacity guarantee. Re-run it
on the actual deployment instance before choosing worker and replica counts.
