"""Export the SpeechBrain ECAPA neural core to ONNX.

The SpeechBrain STFT frontend uses complex operations that are not exported by
the current ONNX path. The resulting model therefore accepts normalized
80-bin FBank features shaped [batch, frames, 80], not raw waveform samples.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import onnx
import torch
from torch import nn

from app.services.model_provider import get_speaker_service


DEFAULT_OUTPUT = ROOT_DIR / "models" / "onnx" / "ecapa_tdnn_voiceprint.onnx"


class EcapaOnnxCore(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.embedding_model = classifier.mods.embedding_model
        self.embedding_normalizer = classifier.mods.mean_var_norm_emb

    def forward(self, features):
        lengths = torch.ones(features.shape[0], device=features.device)
        embedding = self.embedding_model(features, lengths)
        embedding = self.embedding_normalizer(
            embedding,
            torch.ones(embedding.shape[0], device=features.device),
        )
        return embedding.squeeze(1)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = get_speaker_service()
    model = EcapaOnnxCore(service.classifier).eval()
    example_features = torch.zeros(1, 501, 80, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_features,
        str(args.output),
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    print(f"saved: {args.output}")
    print(f"size_mb: {args.output.stat().st_size / (1024 * 1024):.2f}")
    print("input: float32[batch, frames, 80]")
    print("output: float32[batch, 192]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
