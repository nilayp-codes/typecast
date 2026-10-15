"""Pydantic schemas: JobPosting, Skill, Location and their enums."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Seniority(StrEnum):
    """Seniority level of the role."""

    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"


class WorkMode(StrEnum):
    """Where the work happens."""

    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"
    unspecified = "unspecified"


class StrictModel(BaseModel):
    """Base for every schema: unknown fields fail, strings are stripped, no type coercion."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class Skill(StrictModel):
    """One skill mentioned in the job description."""

    name: str = Field(
        min_length=1,
        max_length=40,
        description="Canonical skill name, e.g. 'PostgreSQL', 'Kubernetes'. One skill per entry.",
    )
    required: bool = Field(
        description="True if the posting says the skill is required; false if nice-to-have or preferred.",
    )


class Location(StrictModel):
    """Where the job is based."""

    city: str | None = Field(
        description="City name, or null if no city is stated.",
    )
    country: str | None = Field(
        description="Country name, or null if no country is stated.",
    )
    mode: WorkMode = Field(
        description="Work arrangement. Use 'unspecified' if the posting does not say.",
    )


class JobPosting(StrictModel):
    """Structured fields extracted from one job description."""

    title: str = Field(
        min_length=2,
        max_length=100,
        description="Job title as written in the posting, without company or location.",
    )
    company: str | None = Field(
        description="Hiring company name, or null if not stated.",
    )
    seniority: Seniority = Field(
        description=(
            "Seniority level. Decide from the job title first. If the title gives no level, "
            "use the years of experience required: 0 -> junior, 2-5 -> mid, 5+ -> senior. "
            "Lead, Staff or Principal -> lead."
        ),
    )
    min_years_experience: int | None = Field(
        ge=0,
        le=30,
        description="Minimum years of experience required, or null if not stated.",
    )
    skills: list[Skill] = Field(
        min_length=1,
        max_length=30,
        description="Distinct skills mentioned. No duplicate names.",
    )
    location: Location = Field(
        description="Primary location of the job. If several locations are listed, use the first listed.",
    )
    salary_mentioned: bool = Field(
        description="True if the posting states any salary, pay range or compensation figure.",
    )

    @field_validator("skills")
    @classmethod
    def skills_unique(cls, skills: list[Skill]) -> list[Skill]:
        """Reject skills whose names match after stripping and casefolding."""
        groups: dict[str, list[str]] = {}
        for skill in skills:
            groups.setdefault(skill.name.strip().casefold(), []).append(skill.name)
        dupes = [names for names in groups.values() if len(names) > 1]
        if dupes:
            rendered = "; ".join(", ".join(repr(n) for n in names) for names in dupes)
            raise ValueError(
                f"duplicate skill names: {rendered}. Merge each group into one entry"
            )
        return skills

    @model_validator(mode="after")
    def intern_experience(self) -> Self:
        """Interns cannot require prior years of experience."""
        years = self.min_years_experience
        if self.seniority is Seniority.intern and years not in (None, 0):
            raise ValueError(
                f"seniority is 'intern' but min_years_experience is {years}; "
                "set min_years_experience to null or 0, or choose a different seniority"
            )
        return self
