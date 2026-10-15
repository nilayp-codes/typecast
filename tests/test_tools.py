"""normalize_skill, classify_location, run_tool, postprocess. No LLM involved: the tools are deterministic."""

import json
import logging
from typing import Any

import pytest

from soa import tools
from soa.schemas import JobPosting, Location, Skill, WorkMode
from soa.tools import (
    CLASSIFY_LOCATION,
    NORMALIZE_SKILL,
    SKILL_ALIAS_GROUPS,
    SKILL_ALIASES,
    ClassifyLocationArgs,
    Disagreement,
    NormalizeSkillArgs,
    ToolError,
    ToolResponseError,
    ToolSpec,
    classify_location,
    fill_location,
    jd_header,
    location_disagreements,
    merge_skills,
    normalize_skill,
    postprocess,
    run_tool,
)

HEADER = "Software Engineer\nAcme\nBengaluru, Karnataka, India\nOn-site\nFull-time\n"


def posting(**overrides: Any) -> JobPosting:
    base: dict[str, Any] = {
        "title": "Backend Engineer",
        "company": None,
        "seniority": "mid",
        "min_years_experience": 3,
        "skills": [{"name": "Python", "required": True}],
        "location": {"city": None, "country": None, "mode": "unspecified"},
        "salary_mentioned": False,
    }
    return JobPosting.model_validate_json(json.dumps({**base, **overrides}))


def loc(text: str) -> Location:
    return classify_location(ClassifyLocationArgs(raw=jd_header(text)))


# --- normalize_skill ------------------------------------------------------


def test_normalize_skill_maps_aliases_case_and_whitespace_insensitively() -> None:
    for raw, want in [("k8s", "Kubernetes"), ("  Postgres ", "PostgreSQL"), ("JS", "JavaScript"), ("node  js", "Node.js")]:
        assert normalize_skill(NormalizeSkillArgs(raw=raw, required=True)) == Skill(name=want, required=True)


def test_normalize_skill_keeps_unknown_names_and_the_required_flag() -> None:
    result = normalize_skill(NormalizeSkillArgs(raw="Rust  lang", required=False))
    assert result == Skill(name="Rust lang", required=False)


def test_alias_map_sanity() -> None:
    assert 35 <= len(SKILL_ALIAS_GROUPS) <= 45
    for canonical in SKILL_ALIAS_GROUPS:
        assert SKILL_ALIASES[canonical.casefold()] == canonical  # canonical maps to itself
        assert len(canonical) <= 40  # fits Skill.name
        assert normalize_skill(NormalizeSkillArgs(raw=canonical, required=True)).name == canonical  # idempotent
    for key in SKILL_ALIASES:
        assert key == " ".join(key.split()).casefold()  # keys are stored normalized
    assert set(SKILL_ALIASES.values()) == set(SKILL_ALIAS_GROUPS)


def test_ambiguous_abbreviations_are_not_aliases() -> None:
    # "tf" is often Terraform, "dl" and "py" are too vague: unknown names must pass through unchanged
    for raw in ("tf", "dl", "py"):
        assert raw not in SKILL_ALIASES
        assert normalize_skill(NormalizeSkillArgs(raw=raw, required=True)).name == raw


def test_alias_conflict_fails_loudly() -> None:
    with pytest.raises(ValueError, match="maps to both"):
        tools._build_alias_map({"A": ("x",), "B": ("X",)})


# --- classify_location ----------------------------------------------------


def test_classify_location_reads_place_and_mode_from_header() -> None:
    assert loc(HEADER) == Location(city="Bengaluru", country="India", mode=WorkMode.onsite)


def test_classify_location_country_only_and_trailing_mode() -> None:
    assert loc("AI Engineer\nTrnDigital\nIndia\nRemote\n") == Location(city=None, country="India", mode=WorkMode.remote)
    assert loc("Dev\nCo\nPune, Maharashtra, India (Hybrid)\n") == Location(
        city="Pune", country="India", mode=WorkMode.hybrid
    )


def test_classify_location_metro_area_lines() -> None:
    assert loc("AI Engineer\nBT Group\nGreater Bengaluru Area\nHybrid\n").city == "Bengaluru"
    assert loc("Intern\nLexsi\nMumbai Metropolitan Region\nRemote\n").city == "Mumbai"


def test_classify_location_nothing_found_is_null_and_unspecified() -> None:
    assert loc("Engineer\nAcme\nWe build things.\n") == Location(city=None, country=None, mode=WorkMode.unspecified)


def test_title_words_and_conflicting_labels_do_not_set_mode() -> None:
    assert loc("Remote Sensing Engineer\nAcme\nIndia\n").mode is WorkMode.unspecified
    assert loc("Engineer\nAcme\nIndia\nRemote\nOn-site\n").mode is WorkMode.unspecified


def test_jd_header_is_the_first_ten_non_empty_lines() -> None:
    text = "\n".join(f"line {i}\n" for i in range(1, 30))
    assert jd_header(text).splitlines() == [f"line {i}" for i in range(1, 11)]


# --- run_tool: args and responses ----------------------------------------


def test_run_tool_valid_args_returns_the_response_model() -> None:
    result = run_tool(NORMALIZE_SKILL, {"raw": "k8s", "required": False})
    assert result == Skill(name="Kubernetes", required=False)
    assert isinstance(run_tool(CLASSIFY_LOCATION, {"raw": HEADER}), Location)


@pytest.mark.parametrize(
    "raw_args",
    [
        {"raw": "", "required": True},  # empty
        {"raw": "x" * 41, "required": True},  # longer than Skill.name allows
        {"raw": "k8s", "required": "yes"},  # strict: no coercion
        {"raw": "k8s", "required": True, "extra": 1},  # unknown field
        {"required": True},  # missing
    ],
)
def test_run_tool_invalid_args_return_a_logged_tool_error(
    raw_args: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="soa.tools"):
        result = run_tool(NORMALIZE_SKILL, raw_args)
    assert isinstance(result, ToolError)
    assert result.tool == "normalize_skill" and result.message
    assert "invalid arguments" in caplog.text and result.message in caplog.text


def test_run_tool_bad_response_raises_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    def broken(args: NormalizeSkillArgs) -> Skill:
        return Skill.model_construct(name="", required=args.required)  # bypasses validation: an empty name

    spec = ToolSpec("normalize_skill", NormalizeSkillArgs, Skill, broken)
    with caplog.at_level(logging.ERROR, logger="soa.tools"), pytest.raises(ToolResponseError) as caught:
        run_tool(spec, {"raw": "k8s", "required": True})
    assert caught.value.tool == "normalize_skill"
    assert "name" in caught.value.message
    assert "invalid response" in caplog.text


def test_run_tool_response_of_the_wrong_model_raises() -> None:
    spec = ToolSpec("normalize_skill", NormalizeSkillArgs, Skill, lambda args: Location(city=None, country=None, mode=WorkMode.remote))
    with pytest.raises(ToolResponseError):
        run_tool(spec, {"raw": "k8s", "required": True})


# --- merge, fill-only, postprocess ---------------------------------------


def test_duplicates_created_by_normalizing_are_merged_with_required_any() -> None:
    out = postprocess(
        posting(skills=[
            {"name": "postgres", "required": False},
            {"name": "k8s", "required": False},
            {"name": "PostgreSQL", "required": True},
        ]),
        HEADER,
    )
    assert out.posting.skills == [Skill(name="PostgreSQL", required=True), Skill(name="Kubernetes", required=False)]


def test_merge_skills_keeps_first_position_and_order() -> None:
    merged = merge_skills([Skill(name="A", required=False), Skill(name="B", required=True), Skill(name="a", required=True)])
    assert merged == [Skill(name="A", required=True), Skill(name="B", required=True)]


def test_fill_only_fills_gaps_and_never_overrides() -> None:
    tool = Location(city="Bengaluru", country="India", mode=WorkMode.onsite)
    model = Location(city="Pune", country="India", mode=WorkMode.hybrid)
    assert fill_location(model, tool) == model  # nothing overridden
    assert fill_location(Location(city=None, country=None, mode=WorkMode.unspecified), tool) == tool
    partly = Location(city=None, country="India", mode=WorkMode.hybrid)
    assert fill_location(partly, tool) == Location(city="Bengaluru", country="India", mode=WorkMode.hybrid)


def test_disagreements_only_where_both_sides_have_a_differing_value() -> None:
    tool = Location(city="Bengaluru", country="india", mode=WorkMode.onsite)
    model = Location(city="Pune", country="India", mode=WorkMode.hybrid)
    assert location_disagreements(model, tool) == [
        Disagreement(field="location.city", model_value="Pune", tool_value="Bengaluru"),
        Disagreement(field="location.mode", model_value="hybrid", tool_value="onsite"),
    ]  # country differs only by case: not a disagreement
    empty = Location(city=None, country=None, mode=WorkMode.unspecified)
    assert location_disagreements(empty, tool) == []  # the model abstained: that is a fill, not a disagreement
    assert location_disagreements(model, empty) == []  # the tool abstained


def test_postprocess_returns_and_logs_disagreements_without_changing_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    given = posting(location={"city": "Pune", "country": None, "mode": "hybrid"})
    with caplog.at_level(logging.WARNING, logger="soa.tools"):
        out = postprocess(given, HEADER, doc_id="jd_007")
    assert out.posting.location == Location(city="Pune", country="India", mode=WorkMode.hybrid)
    assert [(d.field, d.model_value, d.tool_value) for d in out.disagreements] == [
        ("location.city", "Pune", "Bengaluru"),
        ("location.mode", "hybrid", "onsite"),
    ]
    assert "jd_007: location.city disagreement: model 'Pune', header tool 'Bengaluru' (model value kept)" in caplog.text
    assert "jd_007: location.mode disagreement: model 'hybrid', header tool 'onsite'" in caplog.text


def test_postprocess_fills_a_null_location_from_the_header() -> None:
    out = postprocess(posting(), HEADER)
    assert out.posting.location == Location(city="Bengaluru", country="India", mode=WorkMode.onsite)
    assert out.disagreements == []


def test_remote_only_in_the_body_does_not_change_mode() -> None:
    # jd_007 style: no mode label in the header window; "Remote work options" is a body bullet (and a bare
    # "Remote" line sits beyond the first ten lines)
    body = "\n".join(["Entry level role for freshers.", "Key Requirements", "* Remote work options"] + ["filler"] * 8 + ["Remote"])
    text = "AI Engineer\nSkyTrainings\nBengaluru, Karnataka, India\n" + body
    assert postprocess(posting(), text).posting.location.mode is WorkMode.unspecified
    onsite = postprocess(posting(location={"city": None, "country": None, "mode": "onsite"}), text)
    assert onsite.posting.location.mode is WorkMode.onsite  # the model's value stays


def test_unusable_header_leaves_the_model_location_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="soa.tools"):
        out = postprocess(posting(location={"city": "Pune", "country": None, "mode": "hybrid"}), "")
    assert out.posting.location == Location(city="Pune", country=None, mode=WorkMode.hybrid)
    assert out.disagreements == []
    assert "invalid arguments" in caplog.text


def test_postprocess_failing_revalidation_raises_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(tools, "merge_skills", lambda skills: [])  # skills min_length=1 -> invalid
    with caplog.at_level(logging.ERROR, logger="soa.tools"), pytest.raises(ToolResponseError) as caught:
        postprocess(posting(), HEADER, doc_id="jd_007")
    assert caught.value.tool == "postprocess"
    assert "skills" in caught.value.message
    assert "jd_007: post-processed posting failed validation" in caplog.text
