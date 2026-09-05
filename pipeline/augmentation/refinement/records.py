"""Domain records for CLIP-guided image replacement."""
from dataclasses import dataclass
from pathlib import Path
from pipeline.augmentation.records import ExtendedSample, GenerationJob


@dataclass(frozen=True)
class RefinementAttempt:
    round_number: int
    seed: int
    description: str
    prompt: str
    candidate_path: str
    score: float

    @classmethod
    def from_dict(cls, source):
        return cls(**source)

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class RefinementTask:
    record: ExtendedSample
    original_score: float
    current_description: str
    current_prompt: str
    round_number: int
    candidate_path: str
    attempts: tuple[RefinementAttempt, ...] = ()

    @property
    def sample_id(self):
        return self.record.sample.sample_id

    @property
    def seed(self):
        return self.record.sample.generation_seed

    def generation_job(self):
        return GenerationJob(self.record, self.current_prompt, Path(self.candidate_path),
                             Path("/dev/null"), True)

    def to_dict(self):
        return dict(record=self.record.to_dict(), original_score=self.original_score,
            current_description=self.current_description, current_prompt=self.current_prompt,
            round_number=self.round_number, candidate_path=self.candidate_path,
            attempts=[attempt.to_dict() for attempt in self.attempts])

    @classmethod
    def from_dict(cls, source):
        return cls(record=ExtendedSample.from_dict(source["record"]),
            original_score=float(source["original_score"]),
            current_description=source["current_description"],
            current_prompt=source["current_prompt"],
            round_number=int(source["round_number"]),
            candidate_path=source["candidate_path"],
            attempts=tuple(RefinementAttempt.from_dict(value) for value in source.get("attempts", [])))


@dataclass(frozen=True)
class CandidateResult:
    sample_id: str
    attempt: RefinementAttempt

    @classmethod
    def from_dict(cls, source):
        return cls(source["sample_id"], RefinementAttempt.from_dict(source["attempt"]))

    def to_dict(self):
        return dict(sample_id=self.sample_id, attempt=self.attempt.to_dict())
