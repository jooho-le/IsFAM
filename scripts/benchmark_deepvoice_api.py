"""Measure deepvoice API latency and throughput at several concurrency levels."""

from argparse import ArgumentParser
import asyncio
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import httpx


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT_DIR / "datasets" / "eval" / "family_real" / "daughter_register_01.m4a"
DEFAULT_REPORT = ROOT_DIR / "reports" / "deepvoice_load_test.md"


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value))
    return ordered[index]


async def run_level(
    client: httpx.AsyncClient,
    url: str,
    audio_name: str,
    audio_bytes: bytes,
    concurrency: int,
    request_count: int,
    expected_signature: tuple[bool, float, int],
) -> dict[str, float | int]:
    semaphore = asyncio.Semaphore(concurrency)

    async def send_one() -> tuple[int, float, float | None, tuple[bool, float, int] | None]:
        async with semaphore:
            started_at = perf_counter()
            try:
                response = await client.post(
                    url,
                    files={"audio_file": (audio_name, audio_bytes, "application/octet-stream")},
                )
                elapsed_ms = (perf_counter() - started_at) * 1000.0
                payload = response.json() if response.status_code == 200 else {}
                signature = None
                if response.status_code == 200:
                    signature = (
                        bool(payload["is_spoofed"]),
                        float(payload["spoof_score"]),
                        int(payload["analyzed_segments"]),
                    )
                return response.status_code, elapsed_ms, payload.get("processing_time_ms"), signature
            except Exception:
                return 0, (perf_counter() - started_at) * 1000.0, None, None

    wall_started_at = perf_counter()
    results = await asyncio.gather(*(send_one() for _ in range(request_count)))
    wall_seconds = perf_counter() - wall_started_at
    latencies = [result[1] for result in results]
    model_times = [result[2] for result in results if result[2] is not None]
    successes = sum(1 for status_code, _, _, _ in results if status_code == 200)
    result_mismatches = sum(
        1 for status_code, _, _, signature in results
        if status_code == 200 and signature != expected_signature
    )
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "successes": successes,
        "errors": request_count - successes,
        "result_mismatches": result_mismatches,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(successes / wall_seconds, 3) if wall_seconds else 0.0,
        "mean_ms": round(mean(latencies), 2),
        "p50_ms": round(median(latencies), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "max_ms": round(max(latencies), 2),
        "mean_model_ms": round(mean(model_times), 2) if model_times else 0.0,
    }


async def async_main(args) -> int:
    audio_bytes = args.audio.read_bytes()
    url = f"{args.base_url.rstrip('/')}/api/v1/anti-spoofing/detect"
    timeout = httpx.Timeout(args.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        warmup_response = await client.post(
            url,
            files={"audio_file": (args.audio.name, audio_bytes, "application/octet-stream")},
        )
        if warmup_response.status_code != 200:
            raise SystemExit(f"Warm-up request failed: {warmup_response.status_code}")
        warmup_payload = warmup_response.json()
        expected_signature = (
            bool(warmup_payload["is_spoofed"]),
            float(warmup_payload["spoof_score"]),
            int(warmup_payload["analyzed_segments"]),
        )

        rows = []
        for concurrency in args.concurrency:
            request_count = max(concurrency, concurrency * args.requests_per_worker)
            row = await run_level(
                client,
                url,
                args.audio.name,
                audio_bytes,
                concurrency,
                request_count,
                expected_signature,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    report_lines = [
        "# Deepvoice API Load Test",
        "",
        f"- endpoint: `{url}`",
        f"- audio: `{args.audio.name}`",
        f"- audio bytes: {len(audio_bytes)}",
        "- one unreported warm-up request was sent before measurement",
        "",
        f"- expected result signature: `{expected_signature}`",
        "",
        "| concurrency | requests | success | errors | result mismatch | throughput req/s | mean ms | p50 ms | p95 ms | max ms | model mean ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {concurrency} | {requests} | {successes} | {errors} | {result_mismatches} | {throughput_rps} | "
            "{mean_ms} | {p50_ms} | {p95_ms} | {max_ms} | {mean_model_ms} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "This is a local-machine benchmark, not a production capacity guarantee. Re-run it",
            "on the actual deployment instance before choosing worker and replica counts.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"saved: {args.report}")
    return 0


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--requests-per-worker", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
