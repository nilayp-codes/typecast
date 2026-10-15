"""Pydantic schemas: JobPosting, Skill, Location and their enums."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
    """Base for every schema: unknown fields fail, strings are stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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
        description="Seniority level of the role.",
    )
    min_years_experience: int | None = Field(
        default=None,
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
        description="Where the job is based.",
    )
    salary_mentioned: bool = Field(
        description="True if the posting states any salary, pay range or compensation figure.",
    )
