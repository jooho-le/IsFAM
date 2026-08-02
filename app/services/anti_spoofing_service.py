import array
import importlib
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


class AntiSpoofingError(Exception):
    """Raised when anti-spoofing model loading or inference fails."""


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


@dataclass(frozen=True)
class AntiSpoofingResult:
    is_spoofed: bool
    spoof_score: float
    threshold: float
    predicted_label: str
    predicted_score: float
    message: str
    model_name: str
    analyzed_segments: int
    max_spoof_segment_index: int
    segment_seconds: float
    label_scores: list[LabelScore]


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise AntiSpoofingError(
            "PyTorch is not installed. Run: pip install -r requirements.txt"
        ) from exc


def _import_transformers() -> tuple[Any, Any]:
    try:
        transformers = importlib.import_module("transformers")
        return (
            transformers.AutoFeatureExtractor,
            transformers.AutoModelForAudioClassification,
        )
    except Exception as exc:
        raise AntiSpoofingError(
            "transformers is not installed or cannot be loaded. Run: pip install -r requirements.txt"
        ) from exc


class AntiSpoofingService:
    """Hugging Face audio classification based anti-spoofing service."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.anti_spoofing_model_name
        self.threshold = settings.anti_spoofing_threshold
        self.target_sample_rate = settings.target_sample_rate
        self.max_audio_seconds = settings.anti_spoofing_max_audio_seconds
        self.window_seconds = settings.anti_spoofing_window_seconds
        self.hop_seconds = settings.anti_spoofing_hop_seconds
        self.batch_size = settings.anti_spoofing_batch_size
        self.spoof_labels = {
            self._normalize_label(label)
            for label in settings.anti_spoofing_spoof_labels
        }
        self.torch = _import_torch()
        auto_feature_extractor, auto_model = _import_transformers()
        self.device = self._resolve_device(settings.device, self.torch)

        logger.info(
            "Loading anti-spoofing model=%s device=%s",
            self.model_name,
            self.device,
        )
        settings.anti_spoofing_model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = auto_feature_extractor.from_pretrained(
            self.model_name,
            cache_dir=str(settings.anti_spoofing_model_dir),
        )
        self.model = auto_model.from_pretrained(
            self.model_name,
            cache_dir=str(settings.anti_spoofing_model_dir),
        )
        self.model.to(self.device)
        self.model.eval()
        self._is_warmed_up = False
        logger.info("Anti-spoofing model loaded")

    def warm_up(self) -> None:
        """Initialize inference kernels before the first user request."""

        if self._is_warmed_up:
            return
        sample_count = max(1, int(self.target_sample_rate * self.window_seconds))
        silent_segment = [0.0] * sample_count
        self._predict_batch_label_scores(
            [silent_segment] * self.batch_size
        )
        self._is_warmed_up = True
        logger.info("Anti-spoofing model warmed up with batch_size=%s", self.batch_size)

    def detect_file(self, wav_path: Path) -> AntiSpoofingResult:
        """Run real/spoof classification on multiple windows of a normalized wav file."""

        samples = self._load_standard_wav_samples(wav_path)
        max_samples = int(self.target_sample_rate * self.max_audio_seconds)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        segment_results: list[tuple[int, float, LabelScore, list[LabelScore]]] = []

        try:
            segments = self._iter_audio_segments(samples)
            for batch_start in range(0, len(segments), self.batch_size):
                batch = segments[batch_start : batch_start + self.batch_size]
                batch_scores = self._predict_batch_label_scores(batch)
                for batch_index, label_scores in enumerate(batch_scores):
                    segment_index = batch_start + batch_index
                    spoof_score = round(
                        sum(
                            label_score.score
                            for label_score in label_scores
                            if self._normalize_label(label_score.label) in self.spoof_labels
                        ),
                        4,
                    )
                    predicted = max(label_scores, key=lambda label_score: label_score.score)
                    segment_results.append((segment_index, spoof_score, predicted, label_scores))
        except Exception as exc:
            logger.exception("Failed to run anti-spoofing inference: %s", wav_path)
            raise AntiSpoofingError("failed to run anti-spoofing inference") from exc

        if not segment_results:
            raise AntiSpoofingError("audio for anti-spoofing has no analyzable segments")

        max_segment_index, spoof_score, predicted, label_scores = max(
            segment_results,
            key=lambda result: result[1],
        )
        is_spoofed = spoof_score >= self.threshold
        message = "spoof" if is_spoofed else "bonafide"

        return AntiSpoofingResult(
            is_spoofed=is_spoofed,
            spoof_score=spoof_score,
            threshold=self.threshold,
            predicted_label=predicted.label,
            predicted_score=predicted.score,
            message=message,
            model_name=self.model_name,
            analyzed_segments=len(segment_results),
            max_spoof_segment_index=max_segment_index,
            segment_seconds=self.window_seconds,
            label_scores=label_scores,
        )

    def _predict_batch_label_scores(
        self,
        segments: list[list[float]],
    ) -> list[list[LabelScore]]:
        inputs = self.feature_extractor(
            segments,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = self.torch.softmax(outputs.logits, dim=-1).detach().cpu()

        return [self._build_label_scores(row) for row in probabilities]

    def _iter_audio_segments(self, samples: list[float]) -> list[list[float]]:
        """Split audio into overlapping windows and include the tail window."""

        window_size = max(1, int(self.target_sample_rate * self.window_seconds))
        hop_size = max(1, int(self.target_sample_rate * self.hop_seconds))
        min_segment_size = int(self.target_sample_rate * 1.0)

        if len(samples) <= window_size:
            return [samples]

        segments: list[list[float]] = []
        starts: list[int] = []
        start = 0

        while start + min_segment_size <= len(samples):
            starts.append(start)
            if start + window_size >= len(samples):
                break
            start += hop_size

        tail_start = max(0, len(samples) - window_size)
        if tail_start not in starts:
            starts.append(tail_start)

        for segment_start in sorted(starts):
            segment = samples[segment_start : segment_start + window_size]
            if len(segment) >= min_segment_size:
                segments.append(segment)

        return segments

    def _build_label_scores(self, probabilities: Any) -> list[LabelScore]:
        id_to_label = getattr(self.model.config, "id2label", {}) or {}
        label_scores: list[LabelScore] = []

        for index, probability in enumerate(probabilities.tolist()):
            label = str(id_to_label.get(index, f"LABEL_{index}"))
            label_scores.append(
                LabelScore(
                    label=label,
                    score=round(float(probability), 4),
                )
            )

        return label_scores

    def _load_standard_wav_samples(self, wav_path: Path) -> list[float]:
        """Load a 16kHz mono PCM16 wav into float samples in [-1, 1]."""

        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                raw_audio = wav_file.readframes(frame_count)
        except Exception as exc:
            logger.exception("Failed to read wav for anti-spoofing: %s", wav_path)
            raise AntiSpoofingError("failed to read wav for anti-spoofing") from exc

        if frame_count <= 0 or not raw_audio:
            raise AntiSpoofingError("audio for anti-spoofing is empty")
        if sample_rate != self.target_sample_rate:
            raise AntiSpoofingError(
                f"audio sample rate is {sample_rate}, expected {self.target_sample_rate}"
            )
        if sample_width != 2:
            raise AntiSpoofingError(
                f"audio sample width is {sample_width} bytes, expected 2 bytes PCM16"
            )

        samples = array.array("h")
        samples.frombytes(raw_audio)
        if sys.byteorder == "big":
            samples.byteswap()

        if channel_count <= 1:
            return [sample / 32768.0 for sample in samples]

        mono_samples: list[float] = []
        for index in range(0, len(samples), channel_count):
            frame = samples[index : index + channel_count]
            mono_samples.append((sum(frame) / len(frame)) / 32768.0)
        return mono_samples

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _resolve_device(configured_device: str, torch_module: Any) -> str:
        device = configured_device.lower().strip()
        if device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if device == "cuda" and not torch_module.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable. Falling back to CPU.")
            return "cpu"
        return device
