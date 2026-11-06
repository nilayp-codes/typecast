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
        description=(
            "Canonical skill name if known (e.g. 'PostgreSQL', 'Kubernetes'), else the posting's own wording. "
            "One skill per entry. Only named technologies, tools, languages, frameworks, platforms, protocols, "
            "named technical methods (e.g. RAG, LoRA) and CS fundamentals (e.g. data structures and algorithms). "
            "Exclude soft skills, degrees, certifications, spoken languages, industry domains, architecture "
            "concepts and vague entries such as 'Coding' or 'AI'."
        ),
    )
    required: bool = Field(
        description=(
            "True if the skill is in the requirements, the described work or the posting's Skills list. "
            "Alternatives ('A or B') take the status of their section. False if preferred, nice-to-have, "
            "bonus, 'a plus' or optional, if only an example ('e.g.'), or if only in the company's "
            "tech-stack section (include those, as false)."
        ),
    )


class Location(StrictModel):
    """Where the job is based."""

    city: str | None = Field(
        description=(
            "One city -> that city, normalised (e.g. 'Mumbai Metropolitan Region' -> 'Mumbai'). "
            "Several cities in one metro region -> the region name (e.g. 'Delhi NCR'). "
            "Several distinct cities, or no city stated -> null."
        ),
    )
    country: str | None = Field(
        description=(
            "Country name, filled when all places are in one country. "
            "Null if fully remote with no place, or no country is stated."
        ),
    )
    mode: WorkMode = Field(
        description=(
            "Work arrangement from an explicit label, policy or statement (e.g. 'Remote Work Policy', "
            "'work remotely', 'working out of our office'). Onsite plus an explicit remote option -> hybrid. "
            "Directly contradicting labels -> unspecified. Office perks alone are not a mode statement "
            "-> unspecified."
        ),
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
            "Seniority level. Decide from the job title first: Intern -> intern; "
            "Junior/Associate -> junior; Senior/Sr -> senior; Lead/Staff/Principal/Architect -> lead. "
            "If the title gives no level, use level words in the description ('entry level', 'fresher', "
            "'junior/mid-level'; use the lowest level stated). Then use the minimum years required: "
            "0-1 -> junior, 2-4 -> mid, 5+ -> senior. If none of these is stated -> mid."
        ),
    )
    min_years_experience: int | None = Field(
        ge=0,
        le=30,
        description=(
            "Lower bound of the years of experience asked for. The requirements section beats the header. "
            "'No experience required' -> 0. Null if not stated."
        ),
    )
    skills: list[Skill] = Field(
        min_length=1,
        max_length=30,
        description="Distinct skills mentioned. No duplicate names. At most 30: keep required skills first.",
    )
    location: Location = Field(
        description="Primary location and work arrangement of the job. See the field descriptions.",
    )
    salary_mentioned: bool = Field(
        description=(
            "True only for a stated pay figure or range (monthly pay and stipends count). "
            "'Competitive', budgets and company revenue do not count."
        ),
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
