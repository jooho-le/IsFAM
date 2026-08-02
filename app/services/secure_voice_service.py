from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.services.anti_spoofing_service import AntiSpoofingResult, AntiSpoofingService
from app.services.risk_scoring_service import RiskScoringResult, RiskScoringService
from app.services.voiceprint_service import FamilyVerificationResult, VoiceprintService
from app.utils.audio_quality import AudioQualityResult


@dataclass(frozen=True)
class SecureVoiceVerificationResult:
    family_verification: FamilyVerificationResult
    anti_spoofing: AntiSpoofingResult
    risk: RiskScoringResult
    processing_time_ms: float
    family_model_time_ms: float
    anti_spoofing_model_time_ms: float


class SecureVoiceVerificationService:
    """Run both voice models and produce one combined verification result."""

    def __init__(
        self,
        voiceprint_service: VoiceprintService,
        anti_spoofing_service: AntiSpoofingService,
        risk_scoring_service: RiskScoringService,
    ):
        self.voiceprint_service = voiceprint_service
        self.anti_spoofing_service = anti_spoofing_service
        self.risk_scoring_service = risk_scoring_service

    def verify(
        self,
        wav_path: Path,
        audio_quality: AudioQualityResult | None = None,
    ) -> SecureVoiceVerificationResult:
        started_at = perf_counter()

        family_started_at = perf_counter()
        family_result = self.voiceprint_service.verify_family_voice(wav_path)
        family_model_time_ms = self._elapsed_ms(family_started_at)

        spoof_started_at = perf_counter()
        anti_spoofing_result = self.anti_spoofing_service.detect_file(wav_path)
        anti_spoofing_model_time_ms = self._elapsed_ms(spoof_started_at)

        risk_result = self.risk_scoring_service.score_secure_voice(
            family_result=family_result,
            anti_spoofing_result=anti_spoofing_result,
        )
        if audio_quality is not None and not audio_quality.is_analyzable:
            risk_result = self._apply_low_quality_policy(risk_result, audio_quality)

        return SecureVoiceVerificationResult(
            family_verification=family_result,
            anti_spoofing=anti_spoofing_result,
            risk=risk_result,
            processing_time_ms=self._elapsed_ms(started_at),
            family_model_time_ms=family_model_time_ms,
            anti_spoofing_model_time_ms=anti_spoofing_model_time_ms,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000.0, 2)

    @staticmethod
    def _apply_low_quality_policy(
        risk: RiskScoringResult,
        quality: AudioQualityResult,
    ) -> RiskScoringResult:
        """Keep model warnings, but never trust a result based on weak call audio."""

        quality_reason = f"통화 음질이 불안정합니다({quality.message}). 추가 음성으로 재확인이 필요합니다."
        return RiskScoringResult(
            is_trusted=False,
            risk_level="caution",
            risk_score=0.35,
            family_confidence=round(risk.family_confidence * 0.6, 4),
            mismatch_confidence=risk.mismatch_confidence,
            final_decision="more_voice_required",
            reasons=risk.reasons + [quality_reason],
        )
