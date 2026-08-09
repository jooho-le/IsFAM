"""Compare PyTorch and ONNX ECAPA embeddings on current family audio."""

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from time import perf_counter
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
import torch

from app.services.model_provider import get_speaker_service
from app.utils.audio import cleanup_temp_files, convert_audio_to_standard_wav


DEFAULT_MODEL = ROOT_DIR / "models" / "onnx" / "ecapa_tdnn_voiceprint.onnx"
DEFAULT_AUDIO_DIR = ROOT_DIR / "datasets" / "eval" / "family_real"
DEFAULT_REPORT = ROOT_DIR / "reports" / "speaker_onnx_validation.md"


@dataclass(frozen=True)
class Sample:
    path: Path
    speaker: str
    pytorch_embedding: np.ndarray
    onnx_embedding: np.ndarray
    pytorch_ms: float
    feature_ms: float
    onnx_ms: float


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", type=float, default=0.65)
    return parser.parse_args()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def speaker_name(path: Path) -> str:
    return path.stem.split("_", 1)[0].lower()


def extract_features(service, wav_path: Path) -> np.ndarray:
    waveform = service._load_standard_wav(wav_path).to(service.device)
    lengths = torch.ones(waveform.shape[0], device=service.device)
    with torch.inference_mode():
        features = service.classifier.mods.compute_features(waveform)
        features = service.classifier.mods.mean_var_norm(features, lengths)
    return features.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"ONNX model does not exist: {args.model}")

    audio_paths = sorted(
        path
        for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a"}
    )
    if len(audio_paths) < 2:
        raise SystemExit(f"Need at least two audio files in {args.audio_dir}")

    service = get_speaker_service()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    samples: list[Sample] = []

    for index, path in enumerate(audio_paths, start=1):
        wav_path = None
        try:
            wav_path = convert_audio_to_standard_wav(path, 16000, 1.0)

            started = perf_counter()
            pytorch_embedding = service.extract_embedding(wav_path).numpy()
            pytorch_ms = (perf_counter() - started) * 1000.0

            started = perf_counter()
            features = extract_features(service, wav_path)
            feature_ms = (perf_counter() - started) * 1000.0

            started = perf_counter()
            onnx_embedding = session.run(None, {"features": features})[0][0]
            onnx_ms = (perf_counter() - started) * 1000.0

            samples.append(
                Sample(
                    path=path,
                    speaker=speaker_name(path),
                    pytorch_embedding=pytorch_embedding,
                    onnx_embedding=onnx_embedding,
                    pytorch_ms=pytorch_ms,
                    feature_ms=feature_ms,
                    onnx_ms=onnx_ms,
                )
            )
            print(f"[{index}/{len(audio_paths)}] {path.name}")
        finally:
            cleanup_temp_files([wav_path])

    max_abs_diffs = [
        float(np.max(np.abs(sample.pytorch_embedding - sample.onnx_embedding)))
        for sample in samples
    ]
    embedding_cosines = [
        cosine(sample.pytorch_embedding, sample.onnx_embedding)
        for sample in samples
    ]

    pair_count = 0
    decision_matches = 0
    pytorch_correct = 0
    onnx_correct = 0
    max_pair_delta = 0.0
    for left, right in combinations(samples, 2):
        expected_same = left.speaker == right.speaker
        pytorch_score = cosine(left.pytorch_embedding, right.pytorch_embedding)
        onnx_score = cosine(left.onnx_embedding, right.onnx_embedding)
        pytorch_same = pytorch_score >= args.threshold
        onnx_same = onnx_score >= args.threshold
        pair_count += 1
        decision_matches += int(pytorch_same == onnx_same)
        pytorch_correct += int(pytorch_same == expected_same)
        onnx_correct += int(onnx_same == expected_same)
        max_pair_delta = max(max_pair_delta, abs(pytorch_score - onnx_score))

    report_lines = [
        "# Speaker ONNX Validation",
        "",
        f"- model: `{args.model.relative_to(ROOT_DIR)}`",
        f"- model size: {args.model.stat().st_size / (1024 * 1024):.2f} MB",
        f"- audio samples: {len(samples)}",
        f"- comparison pairs: {pair_count}",
        f"- threshold: {args.threshold:.2f}",
        f"- mean embedding cosine (PyTorch vs ONNX): {mean(embedding_cosines):.8f}",
        f"- minimum embedding cosine: {min(embedding_cosines):.8f}",
        f"- maximum element absolute difference: {max(max_abs_diffs):.8f}",
        f"- maximum pair similarity delta: {max_pair_delta:.8f}",
        f"- pair decision agreement: {decision_matches}/{pair_count} ({decision_matches / pair_count:.2%})",
        f"- PyTorch pair accuracy: {pytorch_correct}/{pair_count} ({pytorch_correct / pair_count:.2%})",
        f"- ONNX pair accuracy: {onnx_correct}/{pair_count} ({onnx_correct / pair_count:.2%})",
        f"- mean PyTorch full embedding time: {mean(s.pytorch_ms for s in samples):.2f} ms",
        f"- mean feature preprocessing time: {mean(s.feature_ms for s in samples):.2f} ms",
        f"- mean ONNX neural-core time: {mean(s.onnx_ms for s in samples):.2f} ms",
        "",
        "## Scope",
        "",
        "The ONNX model accepts normalized 80-bin FBank features. Android waveform decoding,",
        "16 kHz resampling, FBank extraction, and feature normalization remain to be implemented",
        "and validated before this can be called a complete on-device voice pipeline.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"saved report: {args.report}")
    print(f"decision agreement: {decision_matches}/{pair_count}")
    return 0 if decision_matches == pair_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
