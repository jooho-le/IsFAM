# Speaker ONNX Android Handoff

Build the local handoff bundle after exporting the ONNX model:

```bash
.venv/bin/pip install -r requirements-onnx.txt
HF_HUB_OFFLINE=1 .venv/bin/python scripts/export_speaker_onnx.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/build_speaker_ondevice_bundle.py
```

An optional INT8 candidate can be generated for real-device comparison:

```bash
.venv/bin/python scripts/quantize_speaker_onnx.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_speaker_onnx.py \
  --model models/onnx/ecapa_tdnn_voiceprint_int8.onnx \
  --report reports/speaker_onnx_int8_validation.md
```

The current INT8 experiment reduced the model from 79.61 MB to 20.52 MB and preserved all
45 pair decisions, but changed pair similarity by up to 0.01781 and was slower on the macOS
CPU test. Treat it as an Android benchmark candidate, not the default production model.

Output (intentionally excluded from Git because it contains an 80 MB model):

```text
artifacts/speaker_ondevice_bundle/
  ecapa_tdnn_voiceprint.onnx
  speaker_model_manifest.json
  sine_440hz.pcm
  dual_220_660hz.pcm
  amplitude_ramp_330hz.pcm
```

The manifest contains the model SHA-256, exact input/output contract, preprocessing parameters,
three non-biometric PCM fixtures, expected 192-value embeddings, and expected cosine scores.

The Kotlin implementation is accepted only when:

1. Each PCM fixture hash matches the manifest.
2. Android produces `float32[1, frames, 80]` normalized FBank features.
3. ONNX Runtime produces a 192-value embedding.
4. Android/reference embedding cosine similarity is measured and reported.
5. All reference pair decisions agree before using threshold `0.65` on real family voices.

Do not commit real family voiceprints or generated biometric fixtures. Store production
voiceprints using encryption backed by Android Keystore.
