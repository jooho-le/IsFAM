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


class SpeakerVerificationError(Exception):
    """Raised when speaker embedding extraction or comparison fails."""


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise SpeakerVerificationError(
            "PyTorch is not installed. Run: pip install -r requirements.txt"
        ) from exc


def _import_encoder_classifier() -> Any:
    """Import SpeechBrain EncoderClassifier across supported SpeechBrain versions."""

    import_paths = (
        "speechbrain.inference.classifiers",
        "speechbrain.inference.speaker",
        "speechbrain.pretrained",
    )
    last_error: Exception | None = None

    for module_path in import_paths:
        try:
            module = importlib.import_module(module_path)
            return module.EncoderClassifier
        except Exception as exc:
            last_error = exc

    raise SpeakerVerificationError(
        "SpeechBrain is not installed or EncoderClassifier cannot be loaded. "
        "Run: pip install -r requirements.txt"
    ) from last_error


@dataclass(frozen=True)
class SpeakerComparisonResult:
    similarity: float
    threshold: float
    is_same_speaker: bool
    message: str
    model_name: str


class SpeakerVerificationService:
    """SpeechBrain ECAPA-TDNN based speaker verification service.

    This class is intentionally independent from FastAPI. Later, an
    AntiSpoofingService can be added next to this class and orchestrated from
    the API layer without rewriting the speaker verification model code.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.speaker_model_name
        self.threshold = settings.speaker_threshold
        self.target_sample_rate = settings.target_sample_rate
        self.torch = _import_torch()
        encoder_classifier = _import_encoder_classifier()
        self.device = self._resolve_device(settings.device, self.torch)

        logger.info(
            "Loading speaker verification model=%s device=%s",
            self.model_name,
            self.device,
        )
        settings.speaker_model_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = encoder_classifier.from_hparams(
            source=self.model_name,
            savedir=str(settings.speaker_model_dir),
            run_opts={"device": self.device},
        )
        self._is_warmed_up = False
        logger.info("Speaker verification model loaded")

    def warm_up(self) -> None:
        """Initialize inference kernels before the first user request."""

        if self._is_warmed_up:
            return
        waveform = self.torch.zeros(
            (1, self.target_sample_rate * 3),
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.inference_mode():
            self.classifier.encode_batch(waveform, normalize=True)
        self._is_warmed_up = True
        logger.info("Speaker verification model warmed up")

    def compare_files(self, audio_file_1: Path, audio_file_2: Path) -> SpeakerComparisonResult:
        """Extract embeddings from two audio files and compare them."""

        embedding_1 = self.extract_embedding(audio_file_1)
        embedding_2 = self.extract_embedding(audio_file_2)

        similarity = self.compare_embeddings(embedding_1, embedding_2)

        is_same_speaker = similarity >= self.threshold
        message = "same_speaker" if is_same_speaker else "different_speaker"

        return SpeakerComparisonResult(
            similarity=similarity,
            threshold=self.threshold,
            is_same_speaker=is_same_speaker,
            message=message,
            model_name=self.model_name,
        )

    def compare_embeddings(self, embedding_1: Any, embedding_2: Any) -> float:
        """Return cosine similarity for two speaker embeddings."""

        similarity = self.torch.nn.functional.cosine_similarity(
            embedding_1.flatten().unsqueeze(0),
            embedding_2.flatten().unsqueeze(0),
            dim=1,
        ).item()
        return round(float(similarity), 4)

    def extract_embedding(self, audio_path: Path) -> Any:
        """Load a wav file and return a normalized speaker embedding tensor."""

        waveform = self._load_standard_wav(audio_path).to(self.device)

        try:
            with self.torch.inference_mode():
                # encode_batch expects [batch, time]. Mono waveform is [1, time],
                # so it is treated as a single utterance batch.
                embedding = self.classifier.encode_batch(waveform, normalize=True)
        except Exception as exc:
            logger.exception("Failed to extract speaker embedding: %s", audio_path)
            raise SpeakerVerificationError("failed to extract speaker embedding") from exc

        return embedding.squeeze().detach().cpu().float()

    def _load_standard_wav(self, audio_path: Path) -> Any:
        """Load the generated 16kHz mono PCM wav using only the stdlib wave module."""

        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                raw_audio = wav_file.readframes(frame_count)
        except Exception as exc:
            logger.exception("Failed to read wav for embedding: %s", audio_path)
            raise SpeakerVerificationError("failed to read wav for embedding") from exc

        if frame_count <= 0 or not raw_audio:
            raise SpeakerVerificationError("audio for embedding is empty")

        if sample_rate != self.target_sample_rate:
            raise SpeakerVerificationError(
                f"audio sample rate is {sample_rate}, expected {self.target_sample_rate}"
            )

        if sample_width != 2:
            raise SpeakerVerificationError(
                f"audio sample width is {sample_width} bytes, expected 2 bytes PCM16"
            )

        samples = array.array("h")
        samples.frombytes(raw_audio)
        if sys.byteorder == "big":
            samples.byteswap()

        waveform = self.torch.tensor(samples, dtype=self.torch.float32)

        if channel_count > 1:
            waveform = waveform.reshape(-1, channel_count).mean(dim=1)

        waveform = waveform / 32768.0
        return waveform.unsqueeze(0)

    @staticmethod
    def _resolve_device(configured_device: str, torch_module: Any) -> str:
        """Resolve device string while keeping CPU as a reliable default."""

        device = configured_device.lower().strip()
        if device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"

        if device == "cuda" and not torch_module.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable. Falling back to CPU.")
            return "cpu"

        return device
