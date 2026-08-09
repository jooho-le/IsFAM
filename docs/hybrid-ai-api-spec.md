# IsFAM Hybrid AI Contract

## Target architecture

```text
Android
  - ONNX speaker embedding inference (pending)
  - encrypted family voiceprint storage (pending)
  - local cosine-similarity decision (pending)
  - uploads only the call-audio segment needed for deepvoice detection

FastAPI
  - deepvoice detection (implemented)
  - audio normalization and quality measurement (implemented)
  - batched model inference and startup warm-up (implemented)
```

For a deepvoice-only deployment, disable speaker-model preloading so the server does not spend
memory on the model moved to Android:

```bash
ISFAM_PRELOAD_MODELS=true
ISFAM_PRELOAD_SPEAKER_MODEL=false
```

The Java product server is outside this repository. It may proxy this request later without
changing the audio field or response body below.

## Deepvoice detection API

### Request

```http
POST /api/v1/anti-spoofing/detect
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `audio_file` | file | yes | `wav`, `mp3`, or `m4a`; maximum 25 MB |

The server converts the upload to mono 16 kHz PCM WAV before inference. A 3–5 second speech
segment is recommended. Longer files are split into overlapping windows.

### Success response (`200`)

```json
{
  "analysis_status": "complete",
  "processing_time_ms": 430.25,
  "is_spoofed": false,
  "spoof_score": 0.2817,
  "threshold": 0.5,
  "predicted_label": "real",
  "predicted_score": 0.7183,
  "message": "bonafide",
  "model_name": "Vansh180/deepfake-audio-wav2vec2",
  "analyzed_segments": 2,
  "max_spoof_segment_index": 0,
  "segment_seconds": 5.0,
  "label_scores": [
    {"label": "real", "score": 0.7183},
    {"label": "fake", "score": 0.2817}
  ],
  "audio_quality": {
    "is_analyzable": true,
    "message": "analyzable",
    "duration_seconds": 5.0,
    "rms_energy": 0.08,
    "peak_amplitude": 0.72,
    "speech_ratio": 0.84
  }
}
```

`analysis_status=more_voice_required` means the model was run but the audio was too short,
quiet, or speech-sparse for a reliable final decision. The client should collect another
segment; it must not treat the raw `is_spoofed` value as a confirmed warning in that case.

### Error responses

| Status | Meaning |
|---:|---|
| `400` | Missing or too-short file |
| `413` | File exceeds the configured size limit |
| `415` | Unsupported extension |
| `422` | File cannot be decoded or inspected |
| `500` | Model loading or inference failed |

## Android contract (pending implementation)

The on-device family verifier should expose this logical result to the app UI:

```json
{
  "is_registered_family": true,
  "best_family_id": "local-family-id",
  "similarity": 0.81,
  "threshold": 0.65,
  "model_version": "ecapa-onnx-v1"
}
```

Before this can be implemented, the ECAPA speaker model must be exported to ONNX and verified
against the current PyTorch embeddings. The ONNX asset, Android ONNX Runtime dependency,
Capacitor native bridge, Keystore-backed encryption, and real-device benchmarks are not yet
present in this repository.
