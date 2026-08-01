from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.services.anti_spoofing_service import AntiSpoofingResult, AntiSpoofingService
from app.services.risk_scoring_service import RiskScoringResult, RiskScoringService
from app.services.voiceprint_service import FamilyVerificationResult, VoiceprintService


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

    def verify(self, wav_path: Path) -> SecureVoiceVerificationResult:
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
