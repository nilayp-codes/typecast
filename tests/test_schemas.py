"""Pass and fail cases for every JobPosting validator and notable constraint."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from soa.schemas import JobPosting, Seniority


def posting(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "title": "Backend Engineer",
        "company": "Acme",
        "seniority": "mid",
        "min_years_experience": 3,
        "skills": [{"name": "Python", "required": True}],
        "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
        "salary_mentioned": False,
    }
    data.update(overrides)
    return data


def skills(*names: str) -> list[dict[str, Any]]:
    return [{"name": n, "required": True} for n in names]


def validate(data: dict[str, Any]) -> JobPosting:
    return JobPosting.model_validate_json(json.dumps(data))


def errors(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    with pytest.raises(ValidationError) as exc:
        validate(data)
    return [(".".join(map(str, e["loc"])), e["type"], e["msg"]) for e in exc.value.errors()]


def test_valid_posting() -> None:
    jp = validate(posting())
    assert jp.seniority is Seniority.mid
    assert jp.location.city == "Pune"


# --- skills: no duplicate names -------------------------------------------


def test_distinct_skills_pass() -> None:
    assert len(validate(posting(skills=skills("Python", "PostgreSQL"))).skills) == 2


def test_duplicate_skills_fail_case_and_whitespace_insensitive() -> None:
    [(loc, typ, msg)] = errors(posting(skills=skills(" Python", "python", "SQL", "PYTHON ", "sql")))
    assert (loc, typ) == ("skills", "value_error")
    assert "duplicate skill names: 'Python', 'python', 'PYTHON'; 'SQL', 'sql'" in msg
    assert "Merge each group into one entry" in msg


# --- intern => min_years_experience is None or 0 --------------------------


@pytest.mark.parametrize("years", [None, 0])
def test_intern_with_no_experience_passes(years: int | None) -> None:
    assert validate(posting(seniority="intern", min_years_experience=years)).seniority == "intern"


def test_intern_with_experience_fails() -> None:
    [(loc, typ, msg)] = errors(posting(seniority="intern", min_years_experience=2))
    assert (loc, typ) == ("", "value_error")
    assert "seniority is 'intern' but min_years_experience is 2" in msg


def test_intern_rule_hidden_while_field_errors_exist() -> None:
    # mode="after" model validators only run once every field is valid.
    errs = errors(posting(seniority="intern", min_years_experience=2, skills=skills("Go", "go")))
    assert [loc for loc, _, _ in errs] == ["skills"]


# --- title length ---------------------------------------------------------


@pytest.mark.parametrize("title", ["QA", "x" * 100])
def test_title_length_bounds_pass(title: str) -> None:
    assert validate(posting(title=title)).title == title


@pytest.mark.parametrize(
    ("title", "typ"),
    [("X", "string_too_short"), ("   ", "string_too_short"), ("x" * 101, "string_too_long")],
)
def test_title_length_bounds_fail(title: str, typ: str) -> None:
    assert [(loc, t) for loc, t, _ in errors(posting(title=title))] == [("title", typ)]


# --- skills min/max -------------------------------------------------------


@pytest.mark.parametrize("count", [1, 30])
def test_skills_count_bounds_pass(count: int) -> None:
    names = [f"skill{i}" for i in range(count)]
    assert len(validate(posting(skills=skills(*names))).skills) == count


@pytest.mark.parametrize(("count", "typ"), [(0, "too_short"), (31, "too_long")])
def test_skills_count_bounds_fail(count: int, typ: str) -> None:
    names = [f"skill{i}" for i in range(count)]
    assert [(loc, t) for loc, t, _ in errors(posting(skills=skills(*names)))] == [("skills", typ)]


def test_skill_name_too_long_fails() -> None:
    errs = errors(posting(skills=skills("x" * 41)))
    assert [(loc, t) for loc, t, _ in errs] == [("skills.0.name", "string_too_long")]


# --- years bounds ---------------------------------------------------------


@pytest.mark.parametrize("years", [None, 0, 30])
def test_years_bounds_pass(years: int | None) -> None:
    assert validate(posting(min_years_experience=years)).min_years_experience == years


def test_years_must_be_present() -> None:
    data = posting()
    del data["min_years_experience"]
    assert [(loc, t) for loc, t, _ in errors(data)] == [("min_years_experience", "missing")]


# --- strict mode: no JSON-lax coercion -------------------------------------


def test_string_for_bool_fails() -> None:
    errs = errors(posting(skills=[{"name": "Go", "required": "yes"}]))
    assert [(loc, t) for loc, t, _ in errs] == [("skills.0.required", "bool_type")]


def test_string_for_bool_fails_on_top_level_field() -> None:
    errs = errors(posting(salary_mentioned="true"))
    assert [(loc, t) for loc, t, _ in errs] == [("salary_mentioned", "bool_type")]


def test_enum_strings_still_validate_from_json() -> None:
    jp = validate(posting(seniority="lead", location={"city": None, "country": None, "mode": "onsite"}))
    assert jp.seniority is Seniority.lead


@pytest.mark.parametrize(
    ("years", "typ"),
    [(-1, "greater_than_equal"), (31, "less_than_equal"), (3.5, "int_type"), ("3", "int_type")],
)
def test_years_bounds_fail(years: float | str, typ: str) -> None:
    errs = errors(posting(min_years_experience=years))
    assert [(loc, t) for loc, t, _ in errs] == [("min_years_experience", typ)]


# --- enums ----------------------------------------------------------------


def test_bad_enum_values_fail() -> None:
    errs = errors(posting(seniority="Senior", location={"city": None, "country": None, "mode": "wfh"}))
    assert [(loc, t) for loc, t, _ in errs] == [("seniority", "enum"), ("location.mode", "enum")]
    assert "Input should be 'intern', 'junior', 'mid', 'senior' or 'lead'" in errs[0][2]


def test_enum_values_are_not_stripped() -> None:
    # str_strip_whitespace applies to str fields only, not StrEnum fields.
    assert [loc for loc, _, _ in errors(posting(seniority=" senior "))] == ["seniority"]


# --- extra fields ---------------------------------------------------------


def test_extra_fields_fail_at_every_level() -> None:
    data = posting(
        remote_ok=True,
        location={"city": "Pune", "country": "India", "mode": "hybrid", "state": "MH"},
        skills=[{"name": "Go", "required": True, "years": 2}],
    )
    errs = errors(data)
    assert {(loc, t) for loc, t, _ in errs} == {
        ("remote_ok", "extra_forbidden"),
        ("location.state", "extra_forbidden"),
        ("skills.0.years", "extra_forbidden"),
    }


# --- required-but-nullable ------------------------------------------------


def test_nullable_fields_accept_explicit_null() -> None:
    jp = validate(posting(company=None, location={"city": None, "country": None, "mode": "remote"}))
    assert (jp.company, jp.location.city, jp.location.country) == (None, None, None)


@pytest.mark.parametrize(("path", "field"), [((), "company"), (("location",), "city")])
def test_nullable_fields_must_be_present(path: tuple[str, ...], field: str) -> None:
    data = posting()
    target = data[path[0]] if path else data
    del target[field]
    assert [(loc, t) for loc, t, _ in errors(data)] == [(".".join((*path, field)), "missing")]


# --- whitespace stripping -------------------------------------------------


def test_strings_are_stripped() -> None:
    jp = validate(posting(title="  Backend Engineer  ", skills=skills(" Go ")))
    assert (jp.title, jp.skills[0].name) == ("Backend Engineer", "Go")
