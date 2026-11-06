"""Small, shared JSON contracts for the three reasoning roles."""

from __future__ import annotations


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
PLAN = obj({"questions": STRINGS, "focus": STRINGS})
ANALYSIS = obj({
    "title": STRING,
    "summary": STRING,
    "findings": {"type": "array", "items": obj({
        "claim": STRING, "source_ids": STRINGS,
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    })},
    "risks": STRINGS,
    "limitations": STRINGS,
})
REVIEW = obj({"approved": {"type": "boolean"}, "issues": STRINGS})


class ValidationError(ValueError):
    pass


def validate(value, schema: dict, path: str = "output") -> None:
    """Validate the deliberately small schema subset used above, even for demo models."""
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str, "boolean": bool}[kind]
    if type(value) is not expected:
        raise ValidationError(f"{path} must be {kind}")
    if kind == "object":
        if set(value) != set(schema["required"]):
            raise ValidationError(f"{path} has missing or unexpected fields")
        for key, sub in schema["properties"].items():
            validate(value[key], sub, f"{path}.{key}")
    elif kind == "array":
        if len(value) > 50:
            raise ValidationError(f"{path} has too many entries")
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")
    elif kind == "string" and (not value.strip() or len(value) > 16000):
        raise ValidationError(f"{path} must be nonempty and at most 16000 characters")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path} has an unsupported value")


def validate_analysis(analysis: dict, sources: list[dict]) -> None:
    validate(analysis, ANALYSIS)
    ids = {s["id"] for s in sources if s.get("text", "").strip()}
    if not analysis["findings"]:
        raise ValidationError("Analysis requires at least one supported finding")
    for finding in analysis["findings"]:
        if not finding["source_ids"] or not set(finding["source_ids"]).issubset(ids):
            raise ValidationError("Every finding must cite available, nonempty sources")
