"""Output contract for the job-fit analysis.

Java analogy: a record + Bean Validation. The JSON schema sent to the API guarantees the
*shape* (fields and types). Rules the grammar cannot enforce (ranges, sizes, cross-field
checks) are validated here, on our side, after the response arrives.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Seniority(StrEnum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"


class SkillEvidence(BaseModel):
    skill: str
    evidence: str = Field(description="Short fact from the resume that proves the skill.")


class JobFit(BaseModel):
    role_title: str
    seniority_required: Seniority
    match_score: int = Field(ge=0, le=100, description="0 = no fit, 100 = perfect fit.")
    matched: list[SkillEvidence] = Field(max_length=15)
    missing: list[str] = Field(
        max_length=15, description="Requirements with no evidence in the resume."
    )
    summary: str = Field(max_length=800)

    @model_validator(mode="after")
    def skill_cannot_be_matched_and_missing(self) -> JobFit:
        # Same idea as a class-level @AssertTrue: a rule across two fields.
        matched = {m.skill.casefold() for m in self.matched}
        both = matched & {s.casefold() for s in self.missing}
        if both:
            raise ValueError(f"Skills both matched and missing: {sorted(both)}")
        return self
