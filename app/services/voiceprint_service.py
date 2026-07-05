import array
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.repositories.family_repository import FamilyMemberRecord, FamilyRepository
from app.services.speaker_service import SpeakerVerificationService


@dataclass(frozen=True)
class RegisteredVoiceprint:
    family_id: int
    name: str
    relation: str
    model_name: str


@dataclass(frozen=True)
class FamilyVoiceMatch:
    family_id: int
    name: str
    relation: str
    similarity: float
    sample_count: int = 1
    max_similarity: float | None = None
    mean_similarity: float | None = None
    median_similarity: float | None = None


@dataclass(frozen=True)
class FamilyVerificationResult:
    is_registered_family: bool
    best_match: FamilyVoiceMatch | None
    threshold: float
    candidates: list[FamilyVoiceMatch]
    message: str
    model_name: str


class VoiceprintService:
    """Application service for registering and reading family voiceprints."""

    def __init__(
        self,
        family_repository: FamilyRepository,
        speaker_service: SpeakerVerificationService | None = None,
    ):
        self.family_repository = family_repository
        self.speaker_service = speaker_service

    def register_family_voice(
        self,
        name: str,
        relation: str,
        wav_path: Path,
    ) -> RegisteredVoiceprint:
        """Extract a speaker embedding and store it as a hidden DB voiceprint."""

        normalized_name = name.strip()
        normalized_relation = relation.strip()
        if not normalized_name:
            raise ValueError("name is required")
        if not normalized_relation:
            raise ValueError("relation is required")
        if self.speaker_service is None:
            raise RuntimeError("speaker service is required to register a voiceprint")

        embedding = self.speaker_service.extract_embedding(wav_path)
        embedding_blob = self.embedding_to_bytes(embedding)

        record = self.family_repository.create(
            name=normalized_name,
            relation=normalized_relation,
            embedding=embedding_blob,
            model_name=self.speaker_service.model_name,
        )

        return self._record_to_voiceprint(record)

    def list_family_members(self) -> list[RegisteredVoiceprint]:
        return [self._record_to_voiceprint(record) for record in self.family_repository.list_all()]

    def get_family_member(self, family_id: int) -> RegisteredVoiceprint | None:
        record = self.family_repository.get(family_id)
        if record is None:
            return None
        return self._record_to_voiceprint(record)

    def delete_family_member(self, family_id: int) -> bool:
        return self.family_repository.delete(family_id)

    def verify_family_voice(self, wav_path: Path) -> FamilyVerificationResult:
        """Compare one new voice against every registered family voiceprint."""

        if self.speaker_service is None:
            raise RuntimeError("speaker service is required to verify a voiceprint")

        records = self.family_repository.list_all()
        if not records:
            return FamilyVerificationResult(
                is_registered_family=False,
                best_match=None,
                threshold=self.speaker_service.threshold,
                candidates=[],
                message="no_registered_family_voiceprints",
                model_name=self.speaker_service.model_name,
            )

        query_embedding = self.speaker_service.extract_embedding(wav_path)
        raw_matches: list[FamilyVoiceMatch] = []

        for record in records:
            stored_embedding = self.embedding_from_bytes(
                embedding_blob=record.embedding,
                torch_module=self.speaker_service.torch,
            )
            similarity = self.speaker_service.compare_embeddings(
                query_embedding,
                stored_embedding,
            )
            raw_matches.append(
                FamilyVoiceMatch(
                    family_id=record.id,
                    name=record.name,
                    relation=record.relation,
                    similarity=similarity,
                )
            )

        candidates = self._aggregate_family_matches(raw_matches)
        best_match = candidates[0]
        is_registered_family = best_match.similarity >= self.speaker_service.threshold
        message = (
            "registered_family_matched"
            if is_registered_family
            else "no_registered_family_match"
        )

        return FamilyVerificationResult(
            is_registered_family=is_registered_family,
            best_match=best_match,
            threshold=self.speaker_service.threshold,
            candidates=candidates,
            message=message,
            model_name=self.speaker_service.model_name,
        )

    @staticmethod
    def _aggregate_family_matches(
        raw_matches: list[FamilyVoiceMatch],
    ) -> list[FamilyVoiceMatch]:
        """Group repeated voice samples for the same family profile.

        A competition demo should not trust a single lucky similarity spike.
        When the same name/relation is registered multiple times, the decision
        score uses the median similarity and still exposes max/mean values for
        explainability.
        """

        grouped: dict[tuple[str, str], list[FamilyVoiceMatch]] = {}
        for match in raw_matches:
            key = (match.name.strip().lower(), match.relation.strip().lower())
            grouped.setdefault(key, []).append(match)

        candidates: list[FamilyVoiceMatch] = []
        for matches in grouped.values():
            similarities = [match.similarity for match in matches]
            representative = min(matches, key=lambda match: match.family_id)
            median_similarity = round(float(median(similarities)), 4)
            candidates.append(
                FamilyVoiceMatch(
                    family_id=representative.family_id,
                    name=representative.name,
                    relation=representative.relation,
                    similarity=median_similarity,
                    sample_count=len(matches),
                    max_similarity=round(max(similarities), 4),
                    mean_similarity=round(float(mean(similarities)), 4),
                    median_similarity=median_similarity,
                )
            )

        candidates.sort(key=lambda candidate: candidate.similarity, reverse=True)
        return candidates

    @staticmethod
    def embedding_to_bytes(embedding: Any) -> bytes:
        """Serialize a 1D float embedding without exposing it through the API."""

        values = array.array("f", [float(value) for value in embedding.flatten().tolist()])
        return values.tobytes()

    @staticmethod
    def embedding_from_bytes(embedding_blob: bytes, torch_module: Any) -> Any:
        """Rebuild an embedding tensor. This will be used by the 3rd phase comparison API."""

        values = array.array("f")
        values.frombytes(embedding_blob)
        return torch_module.tensor(values, dtype=torch_module.float32)

    @staticmethod
    def _record_to_voiceprint(record: FamilyMemberRecord) -> RegisteredVoiceprint:
        return RegisteredVoiceprint(
            family_id=record.id,
            name=record.name,
            relation=record.relation,
            model_name=record.model_name,
        )
