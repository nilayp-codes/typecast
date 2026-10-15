"""Deterministic tools applied after validation: normalize_skill and classify_location.

They are post-processing, not model-called (docs/ARCHITECTURE.md "Tool flow"). Arguments are strict Pydantic models
and the responses are the `Skill` and `Location` schemas.
"""

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from soa.errors import format_validation_errors
from soa.schemas import Location, Skill, StrictModel, WorkMode

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

HEADER_LINES = 10
HEADER_MAX_CHARS = 2000

# canonical name -> aliases (matched case-insensitively, whitespace collapsed). The canonical name matches itself.
SKILL_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "JavaScript": ("js", "java script", "ecmascript"),
    "TypeScript": ("ts",),
    "Python": ("python3", "python 3", "py"),
    "PostgreSQL": ("postgres", "psql", "postgre sql"),
    "MySQL": ("my sql",),
    "MongoDB": ("mongo", "mongo db"),
    "Kubernetes": ("k8s", "kube"),
    "Docker": (),
    "AWS": ("amazon web services",),
    "Google Cloud Platform": ("gcp", "google cloud"),
    "Microsoft Azure": ("azure", "ms azure"),
    "TensorFlow": ("tf", "tensor flow"),
    "PyTorch": ("torch", "py torch"),
    "scikit-learn": ("sklearn", "scikit learn", "scikitlearn"),
    "Node.js": ("node", "nodejs", "node js"),
    "React": ("reactjs", "react.js", "react js"),
    "C++": ("cpp", "c plus plus"),
    "C#": ("csharp", "c sharp"),
    "Go": ("golang", "go lang"),
    "CI/CD": ("ci cd", "cicd", "ci-cd"),
    "REST APIs": ("rest", "restful", "rest api", "restful api", "restful apis"),
    "GraphQL": ("graph ql",),
    "Machine Learning": ("ml", "machine-learning"),
    "Deep Learning": ("dl", "deep-learning"),
    "Natural Language Processing": ("nlp",),
    "Generative AI": ("genai", "gen ai", "gen-ai", "generative artificial intelligence"),
    "Large Language Models": ("llm", "llms", "large language model"),
    "Retrieval-Augmented Generation": ("rag", "retrieval augmented generation"),
    "LangChain": ("lang chain",),
    "Hugging Face": ("huggingface",),
    "Apache Spark": ("spark",),
    "Apache Kafka": ("kafka",),
    "Elasticsearch": ("elastic search",),
    "SQL": ("structured query language",),
    "NoSQL": ("no sql", "no-sql"),
    "FastAPI": ("fast api",),
    "Vertex AI": ("vertexai", "google vertex ai"),
    "Redis": (),
    "Git": (),
    "Linux": (),
}

COUNTRIES = frozenset(
    c.casefold()
    for c in (
        "India", "United States", "USA", "United Kingdom", "UK", "Canada", "Germany", "France", "Netherlands",
        "Ireland", "Singapore", "Australia", "United Arab Emirates", "UAE", "Poland", "Spain", "Sweden",
        "Switzerland", "Japan", "Brazil", "Mexico", "Israel", "Philippines", "Malaysia",
    )
)
MODE_LABELS: dict[str, WorkMode] = {
    "remote": WorkMode.remote,
    "hybrid": WorkMode.hybrid,
    "on-site": WorkMode.onsite,
    "onsite": WorkMode.onsite,
    "on site": WorkMode.onsite,
}
_METRO = re.compile(r"^(?:Greater\s+(?P<a>.+?)\s+Area|(?P<b>.+?)\s+Metropolitan\s+Region)$", re.IGNORECASE)
_MODE_SUFFIX = re.compile(r"\(\s*(remote|hybrid|on-?\s?site)\s*\)\s*$", re.IGNORECASE)


def _build_alias_map(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, names in groups.items():
        for name in (canonical, *names):
            key = _squash(name).casefold()
            if aliases.setdefault(key, canonical) != canonical:
                raise ValueError(f"alias {name!r} maps to both {aliases[key]!r} and {canonical!r}")
    return aliases


def _squash(text: str) -> str:
    return " ".join(text.split())


SKILL_ALIASES: dict[str, str] = _build_alias_map(SKILL_ALIAS_GROUPS)


class NormalizeSkillArgs(StrictModel):
    """`raw` is one skill name; `required` is carried through because the response is a full `Skill`."""

    raw: str = Field(min_length=1, max_length=40, description="Skill name as extracted.")
    required: bool = Field(description="Whether the posting requires the skill.")


class ClassifyLocationArgs(StrictModel):
    """`raw` is the JD header (see `jd_header`)."""

    raw: str = Field(min_length=1, max_length=HEADER_MAX_CHARS, description="First lines of the job description.")


def normalize_skill(args: NormalizeSkillArgs) -> Skill:
    """Map a skill name to its canonical spelling. Unknown names are kept, with whitespace collapsed."""
    name = _squash(args.raw)
    return Skill(name=SKILL_ALIASES.get(name.casefold(), name), required=args.required)


def jd_header(text: str) -> str:
    """The first HEADER_LINES non-empty lines of the source text, where title, company, place and mode live."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:HEADER_LINES])[:HEADER_MAX_CHARS]


def classify_location(args: ClassifyLocationArgs) -> Location:
    """Best-effort city, country and work mode from a JD header. Anything not found is null / 'unspecified'."""
    lines = [line.strip() for line in args.raw.splitlines() if line.strip()]
    city, country = _header_place(lines)
    return Location(city=city, country=country, mode=_header_mode(lines))


def _line_mode(line: str) -> WorkMode | None:
    """A mode label on its own line ("Hybrid") or a trailing "(Hybrid)". Body-style phrases never match."""
    label = MODE_LABELS.get(line.casefold().strip(" .:"))
    if label is not None:
        return label
    suffix = _MODE_SUFFIX.search(line)
    return MODE_LABELS.get(suffix.group(1).casefold()) if suffix else None


def _header_mode(lines: list[str]) -> WorkMode:
    found = {mode for mode in (_line_mode(line) for line in lines) if mode is not None}
    return found.pop() if len(found) == 1 else WorkMode.unspecified


def _header_place(lines: list[str]) -> tuple[str | None, str | None]:
    """First line after the title that is `[City, [State,]] Country` (known country) or a metro-area line."""
    for line in lines[1:]:
        place = _MODE_SUFFIX.sub("", line).strip()
        parts = [p.strip() for p in place.split(",")]
        if len(parts) > 4 or len(place) > 80 or any(ch.isdigit() for ch in place) or _line_mode(place):
            continue
        metro = _METRO.match(place)
        if metro:  # LinkedIn "Greater Bengaluru Area" / "Mumbai Metropolitan Region": a city, no country
            return (metro.group("a") or metro.group("b")), None
        if parts[-1].casefold() in COUNTRIES:
            return (parts[0] if len(parts) > 1 else None), parts[-1]
    return None, None


class ToolError(BaseModel):
    """A tool call whose arguments were invalid. Returned (never raised); the caller keeps its original value."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tool: str
    message: str


@dataclass(frozen=True)
class ToolSpec(Generic[A, R]):
    """A tool with its argument model, response model and implementation."""

    name: str
    args_model: type[A]
    response_model: type[R]
    fn: Callable[[A], R]


NORMALIZE_SKILL = ToolSpec("normalize_skill", NormalizeSkillArgs, Skill, normalize_skill)
CLASSIFY_LOCATION = ToolSpec("classify_location", ClassifyLocationArgs, Location, classify_location)


def run_tool(spec: ToolSpec[A, R], raw_args: Mapping[str, object]) -> R | ToolError:
    """Validate `raw_args` against the tool's argument model, then run the tool.

    Invalid arguments return a `ToolError` (logged), not an exception.
    """
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as exc:
        error = ToolError(tool=spec.name, message=format_validation_errors(exc))
        logger.warning("tool %s: invalid arguments: %s", spec.name, error.message)
        return error
    return spec.fn(args)
