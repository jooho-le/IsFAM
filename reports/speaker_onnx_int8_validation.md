# Speaker ONNX Validation

- model: `models/onnx/ecapa_tdnn_voiceprint_int8.onnx`
- model size: 20.52 MB
- audio samples: 10
- comparison pairs: 45
- threshold: 0.65
- mean embedding cosine (PyTorch vs ONNX): 0.99576674
- minimum embedding cosine: 0.99441087
- maximum element absolute difference: 6.66604424
- maximum pair similarity delta: 0.01780611
- pair decision agreement: 45/45 (100.00%)
- PyTorch pair accuracy: 45/45 (100.00%)
- ONNX pair accuracy: 45/45 (100.00%)
- mean PyTorch full embedding time: 90.35 ms
- mean feature preprocessing time: 10.39 ms
- mean ONNX neural-core time: 337.91 ms

## Scope

The ONNX model accepts normalized 80-bin FBank features. Android waveform decoding,
16 kHz resampling, FBank extraction, and feature normalization remain to be implemented
and validated before this can be called a complete on-device voice pipeline.
