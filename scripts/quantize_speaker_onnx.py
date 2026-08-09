"""Create an optional INT8 ECAPA ONNX candidate for Android benchmarking."""

from argparse import ArgumentParser
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "models" / "onnx" / "ecapa_tdnn_voiceprint.onnx"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "onnx" / "ecapa_tdnn_voiceprint_int8.onnx"


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ONNX model does not exist: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    print(f"saved: {args.output}")
    print(f"fp32_mb: {args.input.stat().st_size / (1024 * 1024):.2f}")
    print(f"int8_mb: {args.output.stat().st_size / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
