import json

import httpx

from finsight.contracts import Prediction, ReasonCode
from modelforge import prompts
from modelforge.adapters.qwen import QwenAdapter
from modelforge.datasets import finqa
from modelforge.runner import evaluate_item, summarize
from modelforge.stats import cluster_bootstrap_mean, paired_difference


def test_cluster_bootstrap_is_deterministic_and_bounded():
    values = {f"q{i}": float(i % 2) for i in range(40)}
    groups = {f"q{i}": f"g{i // 4}" for i in range(40)}
    first = cluster_bootstrap_mean(values, groups)
    assert first == cluster_bootstrap_mean(values, groups)
    mean, low, high = first
    assert low <= mean <= high and mean == 0.5


def test_paired_difference_on_identical_configs_is_zero():
    values = {f"q{i}": 1.0 for i in range(10)}
    groups = {k: "g" for k in values}
    assert paired_difference(values, values, groups)[0] == 0


def test_failures_stay_in_the_denominator(finqa_example):
    item = finqa.to_item(finqa_example, "test")
    predictions = [Prediction(item_id=item.id, config_id="x", reason_code=ReasonCode.MODEL_OUTPUT_INVALID, correct=False)]
    summary = summarize(predictions, [item])
    assert summary["accuracy_answerable"]["mean"] == 0.0
    assert summary["false_abstention_answerable"]["mean"] == 0.0
    assert summary["invalid_output_rate"]["mean"] == 1.0


def test_schema_is_strict_for_every_object():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(prompts.answer_schema())


def _sse(text: str, finish: str = "stop") -> bytes:
    chunks = [{"choices": [{"delta": {"content": text[:10]}}]}, {"choices": [{"delta": {"content": text[10:]}, "finish_reason": finish}]},
              {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 40}}]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def test_qwen_adapter_end_to_end_with_mock_server(finqa_example):
    item = finqa.to_item(finqa_example, "test")
    gold = finqa.gold_answer_output(item).model_dump_json()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(gold), headers={"content-type": "text/event-stream"})

    adapter = QwenAdapter("finsight-lora", transport=httpx.MockTransport(handler))
    prediction = evaluate_item(item, adapter)
    assert prediction.reason_code is ReasonCode.ANSWERED and prediction.correct
    assert prediction.attempts[0].input_tokens_uncached == 120 and prediction.attempts[0].ttft_ms is not None
    body = seen["body"]
    assert body["temperature"] == 0 and body["response_format"]["type"] == "json_schema"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen_truncation_is_operational_not_abstention(finqa_example):
    item = finqa.to_item(finqa_example, "test")
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_sse('{"abstain": tr', "length")))
    prediction = evaluate_item(item, QwenAdapter("m", transport=transport))
    assert prediction.reason_code is ReasonCode.OPERATIONAL
