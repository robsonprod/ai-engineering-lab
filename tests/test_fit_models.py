import json

import pytest
from anthropic.resources.beta.messages.messages import to_json_output_format
from pydantic import ValidationError

from ai_lab.fit import JobFit, Seniority
from tests.conftest import fit_json


def test_valid_output_becomes_typed_object() -> None:
    fit = JobFit.model_validate_json(fit_json())

    assert fit.seniority_required is Seniority.SENIOR
    assert fit.matched[0].skill == "Kafka"


def test_score_out_of_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        JobFit.model_validate_json(fit_json(match_score=120))


def test_skill_cannot_be_matched_and_missing() -> None:
    with pytest.raises(ValidationError, match="both matched and missing"):
        JobFit.model_validate_json(fit_json(missing=["kafka"]))


def test_unknown_seniority_is_rejected() -> None:
    with pytest.raises(ValidationError):
        JobFit.model_validate_json(fit_json(seniority_required="rockstar"))


def test_sdk_moves_constraints_the_api_cannot_enforce_into_descriptions() -> None:
    """Constrained decoding guarantees the shape; ranges are only a hint to the model.

    That is why Pydantic still validates on our side after the response arrives.
    """
    schema = to_json_output_format(JobFit)["schema"]
    score = schema["properties"]["match_score"]

    assert "maximum" not in score
    assert "maximum: 100" in score["description"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"match_score", "matched", "missing"}
    json.dumps(schema)  # serializable as sent on the wire
