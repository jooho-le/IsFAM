# Speaker ONNX Validation

- model: `models/onnx/ecapa_tdnn_voiceprint.onnx`
- model size: 79.61 MB
- audio samples: 10
- comparison pairs: 45
- threshold: 0.65
- mean embedding cosine (PyTorch vs ONNX): 0.99999998
- minimum embedding cosine: 0.99999982
- maximum element absolute difference: 0.00010586
- maximum pair similarity delta: 0.00000045
- pair decision agreement: 45/45 (100.00%)
- PyTorch pair accuracy: 45/45 (100.00%)
- ONNX pair accuracy: 45/45 (100.00%)
- mean PyTorch full embedding time: 82.08 ms
- mean feature preprocessing time: 9.92 ms
- mean ONNX neural-core time: 148.95 ms

## Scope

The ONNX model accepts normalized 80-bin FBank features. Android waveform decoding,
16 kHz resampling, FBank extraction, and feature normalization remain to be implemented
and validated before this can be called a complete on-device voice pipeline.
