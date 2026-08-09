# Deepvoice Concurrency Optimization

Local CPU test using the same 10.9-second audio file and one Uvicorn worker.

| concurrency | before mean | after mean | before p95 | after p95 | before req/s | after req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 510 ms | 503 ms | 510 ms | 503 ms | 1.96 | 1.99 |
| 5 | 2,290 ms | 1,599 ms | 2,712 ms | 2,258 ms | 1.84 | 2.21 |
| 10 | 3,794 ms | 2,505 ms | 5,950 ms | 3,950 ms | 1.68 | 2.53 |
| 20 | 7,017 ms | 4,474 ms | 11,146 ms | 7,768 ms | 1.68 | 2.57 |

At concurrency 20, limiting inference to two worker threads improved throughput by 52.9%,
reduced mean latency by 36.2%, and reduced p95 latency by 30.3%. All 36 measured responses
matched the warm-up result signature (`is_spoofed`, `spoof_score`, and analyzed segment count),
and no HTTP errors occurred.

These numbers are local measurements, not production capacity guarantees. Re-run
`scripts/benchmark_deepvoice_api.py` on the deployment instance before sizing replicas.
