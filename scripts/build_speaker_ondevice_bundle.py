"""Build a shareable, git-ignored Android ONNX handoff bundle."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.model_provider import get_speaker_service


DEFAULT_MODEL = ROOT_DIR / "models" / "onnx" / "ecapa_tdnn_voiceprint.onnx"
DEFAULT_OUTPUT = ROOT_DIR / "artifacts" / "speaker_ondevice_bundle"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_features(service, waveform: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(service.device)
    lengths = torch.ones(1, device=service.device)
    with torch.inference_mode():
        features = service.classifier.mods.compute_features(tensor)
        features = service.classifier.mods.mean_var_norm(features, lengths)
    return features.detach().cpu().numpy().astype(np.float32)


def synthetic_pcm(sample_rate: int) -> dict[str, np.ndarray]:
    seconds = 3
    times = np.arange(sample_rate * seconds, dtype=np.float64) / sample_rate
    signals = {
        "sine_440hz": 0.30 * np.sin(2.0 * np.pi * 440.0 * times),
        "dual_220_660hz": (
            0.20 * np.sin(2.0 * np.pi * 220.0 * times)
            + 0.12 * np.sin(2.0 * np.pi * 660.0 * times)
        ),
        "amplitude_ramp_330hz": (
            np.linspace(0.05, 0.35, sample_rate * seconds)
            * np.sin(2.0 * np.pi * 330.0 * times)
        ),
    }
    return {
        name: np.clip(np.rint(signal * 32767.0), -32768, 32767).astype("<i2")
        for name, signal in signals.items()
    }


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            f"ONNX model does not exist: {args.model}. Run scripts/export_speaker_onnx.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_target = args.output_dir / args.model.name
    shutil.copy2(args.model, model_target)

    service = get_speaker_service()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    references = []
    embeddings: dict[str, np.ndarray] = {}

    for name, pcm in synthetic_pcm(service.target_sample_rate).items():
        pcm_path = args.output_dir / f"{name}.pcm"
        pcm_path.write_bytes(pcm.tobytes())
        waveform = pcm.astype(np.float32) / 32768.0
        features = normalized_features(service, waveform)
        embedding = session.run(None, {"features": features})[0][0].astype(np.float32)
        embeddings[name] = embedding
        references.append(
            {
                "name": name,
                "pcm_file": pcm_path.name,
                "pcm_sha256": file_sha256(pcm_path),
                "sample_count": int(pcm.size),
                "feature_shape": list(features.shape),
                "embedding": [round(float(value), 8) for value in embedding],
            }
        )

    pair_cosines = []
    names = sorted(embeddings)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = embeddings[left_name]
            right = embeddings[right_name]
            score = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            pair_cosines.append(
                {"left": left_name, "right": right_name, "cosine_similarity": round(score, 8)}
            )

    manifest = {
        "format_version": 1,
        "model": {
            "file": model_target.name,
            "sha256": file_sha256(model_target),
            "size_bytes": model_target.stat().st_size,
            "opset": 17,
            "input": {"name": "features", "dtype": "float32", "shape": ["batch", "frames", 80]},
            "output": {"name": "embedding", "dtype": "float32", "shape": ["batch", 192]},
        },
        "preprocessing": {
            "pcm": "signed little-endian 16-bit mono",
            "sample_rate": 16000,
            "window_ms": 25,
            "hop_ms": 10,
            "n_fft": 400,
            "n_mels": 80,
            "feature_normalization": "per-utterance mean; std_norm=false",
        },
        "comparison": {"metric": "cosine_similarity", "current_threshold": 0.65},
        "references": references,
        "reference_pair_cosines": pair_cosines,
    }
    manifest_path = args.output_dir / "speaker_model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"bundle: {args.output_dir}")
    print(f"model_sha256: {manifest['model']['sha256']}")
    print(f"references: {len(references)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
