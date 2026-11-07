"""GPU-side logic that can be tested without a GPU: LoRA target discovery, chat-template rendering,
the recovery-test comparison, serving flags, the comparison manifest, throughput and agreement."""

import json
import re
from pathlib import Path

import httpx
import pytest

from finsight.contracts import ModelAttempt, Prediction, ReasonCode
from modelforge import splits
from modelforge.adapters.qwen import QwenAdapter
from modelforge.probe import discover_targets
from modelforge.runner import _throughput, agreement
from modelforge.serve import build_command, parse_loras
from modelforge.training.recovery_test import compare
from modelforge.training.train_qlora import _render, latest_checkpoint

DENSE = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def test_dense_qwen3_targets_are_a_suffix_list_covering_every_layer():
    names = [f"model.layers.{i}.{m}" for i in range(4) for m in DENSE] + ["lm_head"]
    targets, report = discover_targets(names, expected_layers=4)
    assert targets == sorted(m.rsplit(".", 1)[1] for m in DENSE)
    assert report["coverage_ok"] and report["target_form"] == "suffix_list" and report["decoder_prefix"] == "model"


def test_hybrid_multimodal_targets_use_an_anchored_regex_that_excludes_vision():
    linear_attn = ["linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba", "linear_attn.out_proj"]
    mlp = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    names = [f"model.language_model.layers.{i}.{m}" for i in range(3) for m in linear_attn + mlp]
    names += [f"model.language_model.layers.3.{m}" for m in DENSE]
    names += ["model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.out_proj", "model.visual.merger.mlp.0"]
    targets, report = discover_targets(names, expected_layers=4)
    assert isinstance(targets, str) and report["coverage_ok"]
    assert re.fullmatch(targets, "model.language_model.layers.2.linear_attn.out_proj")
    assert re.fullmatch(targets, "model.language_model.layers.3.self_attn.q_proj")
    assert not re.fullmatch(targets, "model.visual.blocks.0.attn.out_proj")


def test_layer_without_mlp_fails_coverage():
    names = [f"model.layers.{i}.{m}" for i in range(2) for m in DENSE if i == 0 or "mlp" not in m]
    assert discover_targets(names, expected_layers=2)[1]["coverage_ok"] is False


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
                       if m["role"] != "assistant")
        assistant = [m for m in messages if m["role"] == "assistant"]
        if assistant:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + assistant[0]["content"] + "<|im_end|>\n"
        elif add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return text


def test_render_ends_the_completion_exactly_at_eos():
    example = {"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "{}"}]}
    rendered = _render(FakeTokenizer(), example, enable_thinking=False)
    assert rendered["prompt"].endswith("<think>\n\n</think>\n\n")
    assert rendered["completion"] == "{}<|im_end|>"


def _record(steps, lr=None, loss=None, resumed_from=None):
    history = [{"step": s, "loss": (loss or {}).get(s, 1.0 / s), "learning_rate": (lr or {}).get(s, 1e-4 * s)}
               for s in steps]
    return {"log_history": history, "global_step": max(steps), "resumed_from": resumed_from}


FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
         "trainer_state.json"]


def test_recovery_compare_passes_when_state_and_data_position_are_restored():
    hashes = [f"h{i}" for i in range(30)]
    report = compare(_record(range(1, 7)), _record(range(1, 7), resumed_from="x/checkpoint-4"),
                     hashes, hashes[16:24] + ["lookahead"], grad_accum=4, checkpoint_files=FILES)
    assert report["passed"], report


def test_recovery_compare_catches_scheduler_and_data_position_errors():
    hashes = [f"h{i}" for i in range(30)]
    bad_lr = _record(range(1, 7), lr={5: 9.9}, resumed_from="x/checkpoint-4")
    report = compare(_record(range(1, 7)), bad_lr, hashes, hashes[12:20], grad_accum=4, checkpoint_files=FILES[:-1])
    checks = report["checks"]
    assert not checks["learning_rate_identical"] and not checks["data_position_identical"]
    assert not checks["checkpoint_files_present"] and not report["passed"]


def test_latest_checkpoint_ignores_incomplete_ones(tmp_path: Path):
    for step, complete in ((2, True), (4, True), (6, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    assert latest_checkpoint(tmp_path).name == "checkpoint-4"


def test_serve_command_binds_localhost_and_ignores_model_generation_defaults():
    command = build_command("Qwen/Qwen3-4B", "abc", loras=parse_loras(["tuned=artifacts/adapters/x/adapter"]))
    joined = " ".join(command)
    assert "--host 127.0.0.1" in joined and "--generation-config vllm" in joined
    assert "--revision abc" in joined and "--tokenizer-revision abc" in joined
    assert "--enable-lora" in joined and "tuned=artifacts/adapters/x/adapter" in joined
    with pytest.raises(ValueError):
        parse_loras(["missing-path"])
    two = build_command("m", "r", loras=parse_loras(["b=/x/b", "a=/x/a"]))
    assert two.count("--lora-modules") == 1 and two[-2:] == ["a=/x/a", "b=/x/b"]


def test_qwen_request_sets_every_sampling_parameter_and_fits_the_context():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.33.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-4B", "root": "Qwen/Qwen3-4B", "max_model_len": 5120}]})
        seen["body"] = json.loads(request.content)
        return httpx.Response(500)

    adapter = QwenAdapter("Qwen/Qwen3-4B", transport=httpx.MockTransport(handler))
    assert adapter.server_info()["vllm_version"] == "0.33.0"
    from modelforge.items import EvalItem
    item = EvalItem(id="x", dataset="finqa", group="g", company="c", year="2020", question="q", passages=[], rows={},
                    answerable=True, gold_answer="1", gold_steps=None, gold_passage_ids=[], meta={})
    generation = adapter.answer(item, "s", "u", {})
    body = seen["body"]
    assert body["max_tokens"] == 1024 and 3411 + body["max_tokens"] <= 5120
    for key, value in {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0, "presence_penalty": 0,
                       "frequency_penalty": 0, "repetition_penalty": 1, "seed": 0}.items():
        assert body[key] == value
    assert generation.text is None and generation.attempts[0].status == "retryable_error"
    assert "HTTP 500" in generation.attempts[0].error


def test_comparison_manifest_is_derived_frozen_and_paired():
    manifest = splits.build_comparison()
    assert manifest == splits.build_comparison()
    shared, expanded = manifest["samples"]["test_shared_300"], manifest["samples"]["test_expanded_1247"]
    assert len(shared) == 300 and len(expanded) == 1247 and set(shared) <= set(expanded)
    declared = manifest["declared_unanswerable"]["test_shared_300"]
    assert len(declared) == 50 and all(i.endswith("::unanswerable") for i in declared)
    assert manifest["parent"]["sha256"] == splits.load("finqa-grouped")["sha256"]
    committed = Path("evals/manifests/finqa-grouped-comparison.json")
    if committed.exists():
        assert splits.load("finqa-grouped-comparison")["sha256"] == manifest["sha256"]


def test_throughput_comes_from_saved_timing():
    timing = [{"items": 10, "seconds": 5.0, "concurrency": 4}, {"items": 10, "seconds": 5.0, "concurrency": 4}]
    result = _throughput(timing, output_tokens=2000)
    assert result["items_per_second"] == 2.0 and result["output_tokens_per_second"] == 200.0
    assert _throughput([], 0) is None


def test_agreement_measures_rather_than_assumes_identical_reruns(tmp_path: Path):
    def write(run: Path, outputs: dict[str, str]):
        run.mkdir()
        lines = [Prediction(item_id=i, config_id="c", reason_code=ReasonCode.ANSWERED, raw_output=o, executed=o,
                            correct=True).model_dump_json() for i, o in outputs.items()]
        (run / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    write(tmp_path / "a", {"q1": "1", "q2": "2", "q3": "3"})
    write(tmp_path / "b", {"q1": "1", "q2": "2", "q3": "4"})
    result = agreement(tmp_path / "a", tmp_path / "b")
    assert result["common_items"] == 3 and result["identical_outcome"] == round(2 / 3, 4)
    assert result["differing_items"] == ["q3"]


def test_fresh_runs_need_an_empty_folder_and_resume_must_match(tmp_path: Path):
    from modelforge.training.train_qlora import check_run_identity
    out = tmp_path / "run"
    identity = {"max_seq_len": 4096, "train_sha256": "a"}
    check_run_identity(out, identity, resume=False)
    with pytest.raises(SystemExit):
        check_run_identity(out, identity, resume=False)          # a second fresh start in the same folder
    check_run_identity(out, identity, resume=True)                # same experiment: allowed
    with pytest.raises(SystemExit):
        check_run_identity(out, {**identity, "max_seq_len": 3072}, resume=True)


def test_session_totals_include_interrupted_sessions(tmp_path: Path):
    from modelforge.training.train_qlora import session_totals
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "session-1.json").write_text(json.dumps(
        {"status": "running", "wall_seconds": 600.0, "gpu_energy_joules": 1000.0}))
    (tmp_path / "sessions" / "session-2.json").write_text(json.dumps(
        {"status": "completed", "wall_seconds": 300.0, "gpu_energy_joules": 500.0}))
    totals = session_totals(tmp_path)
    assert totals["wall_seconds"] == 900.0 and totals["gpu_energy_joules"] == 1500.0 and totals["complete"] is False


def test_evidence_subcategories_keep_formatting_apart_from_wrong_passages(finqa_example):
    from finsight.contracts import AnswerOutput
    from modelforge.datasets import finqa
    from modelforge.runner import _evidence_subcategories
    item = finqa.to_item(finqa_example, "test")
    table = next(p for p in item.passages if "5,735" in p.text)
    text_id = next(p.id for p in item.passages if p.id.startswith("text_"))
    def output(original, pid):
        return AnswerOutput.model_validate({"abstain": False, "abstain_reason": None, "operands": [
            {"name": "v0", "original_text": original, "value": "5735", "unit": "USD", "scale": "units", "period": None,
             "passage_id": pid}], "program": [{"op": "add", "args": ["v0", "v0"]}], "answer": "11470",
            "answer_unit": None, "claims": []}).model_dump_json()
    wrong_passage = Prediction(item_id=item.id, config_id="c", reason_code=ReasonCode.VALIDATION_CONTRADICTION,
                               raw_output=output("$5,735", text_id),
                               issues=[f"operand v0 ('$5,735') is not supported by passage {text_id}"])
    assert _evidence_subcategories(wrong_passage, item) == {"figure is in a different passage than the one cited"}
    formatting = Prediction(item_id=item.id, config_id="c", reason_code=ReasonCode.VALIDATION_CONTRADICTION,
                            raw_output=output("5735", table.id),
                            issues=[f"operand v0 ('5735') is not supported by passage {table.id}"])
    assert _evidence_subcategories(formatting, item) == {"quote formatting differs from the passage (e.g. added $ or commas)"}
