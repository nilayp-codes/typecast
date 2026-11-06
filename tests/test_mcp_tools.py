"""MCP tool logic (docs/MCP.md section 2): called directly, no MCP SDK needed."""

import json
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from soa.errors import format_retry_feedback
from soa.mcp_tools import MAX_POSTING_CHARS, ValidationResult, get_job_posting_schema, validate_job_posting
from soa.schemas import JobPosting

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": None,
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}],
    "location": {"city": None, "country": None, "mode": "unspecified"},
    "salary_mentioned": False,
}


def check(**changes: Any) -> ValidationResult:
    return validate_job_posting(json.dumps({**VALID, **changes}))


def test_a_valid_posting_returns_the_parsed_posting() -> None:
    result = validate_job_posting(json.dumps(VALID))
    assert result.valid and result.errors == []
    assert result.posting == JobPosting.model_validate_json(json.dumps(VALID))


def test_an_enum_value_given_as_a_plain_string_passes() -> None:
    # strict=True rejects "mid" for a StrEnum under model_validate(dict); JSON-mode validation accepts it.
    with pytest.raises(ValidationError):
        JobPosting.model_validate(VALID)
    result = check(seniority="senior")
    assert result.valid and result.posting is not None and result.posting.seniority == "senior"


def test_malformed_json_keeps_its_error() -> None:
    result = validate_job_posting('```json\n{"title": "x"}\n```')
    assert not result.valid and result.posting is None
    assert len(result.errors) == 1 and result.errors[0].startswith("(root): Invalid JSON: ")


def test_a_json_array_is_not_an_object() -> None:
    assert validate_job_posting("[]").errors == ["(root): Input should be an object (got: [])"]


def test_a_wrong_type_is_reported_with_its_field() -> None:
    result = check(min_years_experience="3")
    assert not result.valid
    assert result.errors == ["min_years_experience: Input should be a valid integer (got: '3')"]


def test_an_unknown_and_a_missing_field_are_both_reported() -> None:
    data = {k: v for k, v in VALID.items() if k != "company"} | {"remote_ok": True}
    result = validate_job_posting(json.dumps(data))
    assert "remote_ok: Extra inputs are not permitted (got: True)" in result.errors
    assert any(line.startswith("company: Field required") for line in result.errors)


def test_more_than_30_skills_gives_the_count_aware_line() -> None:
    skills = [{"name": f"Skill{i}", "required": i < 10} for i in range(39)]
    result = check(skills=skills)
    assert result.errors == ["skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones"]


def test_duplicate_skills_are_rejected() -> None:
    result = check(skills=[{"name": "SQL", "required": True}, {"name": "sql", "required": False}])
    assert len(result.errors) == 1
    assert result.errors[0].startswith("skills: Value error, duplicate skill names: 'SQL', 'sql'. Merge each group")


def test_the_intern_rule_is_reported_at_the_root() -> None:
    result = check(seniority="intern", min_years_experience=2)
    assert len(result.errors) == 1 and result.errors[0].startswith("(root): Value error, seniority is 'intern'")


def test_errors_are_exactly_the_retry_formatters_lines() -> None:
    text = json.dumps({**VALID, "seniority": "wizard", "skills": [{"name": f"S{i}", "required": True} for i in range(31)]})
    with pytest.raises(ValidationError) as raised:
        JobPosting.model_validate_json(text)
    assert validate_job_posting(text).errors == format_retry_feedback(raised.value).splitlines()


def test_a_payload_at_the_limit_is_parsed_and_one_over_is_refused_unparsed() -> None:
    at_limit = json.dumps(VALID) + " " * (MAX_POSTING_CHARS - len(json.dumps(VALID)))
    assert len(at_limit) == MAX_POSTING_CHARS and validate_job_posting(at_limit).valid
    over = validate_job_posting(at_limit + " ")
    assert not over.valid and over.posting is None
    assert over.errors == [f"(root): payload is {MAX_POSTING_CHARS + 1} characters, maximum is {MAX_POSTING_CHARS}"]


def test_an_oversized_payload_is_refused_before_parsing() -> None:
    result = validate_job_posting("{" * 50_000)  # would be a JSON error if it were parsed
    assert result.errors == ["(root): payload is 50000 characters, maximum is 20000"]


def test_the_schema_is_the_models_schema() -> None:
    assert get_job_posting_schema() == JobPosting.model_json_schema()


def test_result_fields_must_agree() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(valid=True, errors=["x"], posting=None)
    with pytest.raises(ValidationError):
        ValidationResult(valid=False, errors=[], posting=None)


def test_importing_the_module_loads_neither_litellm_nor_mcp() -> None:
    code = "import sys, soa.mcp_tools; print(sorted(m for m in sys.modules if m.split('.')[0] in ('litellm', 'mcp')))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
