"""Gemini visual judging and the pinned Docker harness integration, without paid calls."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error

import pytest

from tasks import scienceagentbench_eval_driver as driver
from tasks import task_scienceagentbench as sab


@pytest.fixture
def judge(monkeypatch, tmp_path):
    monkeypatch.setenv("FOH_JUDGE_API_KEY", "fake-google-judge-key")
    monkeypatch.setenv("FOH_JUDGE_MODEL", "gemini-2.5-pro")
    path = Path(sab.__file__).with_name("scienceagentbench_visual_judge.py")
    spec = importlib.util.spec_from_file_location("isolated_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "FOH_JUDGE_API_KEY" not in os.environ
    original = tmp_path / "original.py"
    original.write_text('raise RuntimeError("must not execute upstream client init")\nPROMPT_ORIGIN = "Compare generated then gold."\n')
    module.ORIGINAL_JUDGE = original
    return module


def test_native_judge_sends_images_and_averages_three_scores(judge, monkeypatch):
    texts = iter(["[FINAL SCORE]: 90", "[FINAL SCORE]: 60", "unparseable"])
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        assert request.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        assert request.get_header("X-goog-api-key") == "fake-google-judge-key"
        assert "fake-google-judge-key" not in request.full_url
        body = json.loads(request.data)
        assert body["contents"][0]["parts"] == [
            {"text": "Compare generated then gold."},
            {"inlineData": {"mimeType": "image/png", "data": "pred-image"}},
            {"inlineData": {"mimeType": "image/png", "data": "gold-image"}},
        ]
        return io.StringIO(json.dumps({"candidates": [{"content": {"parts": [
            {"text": "[FINAL SCORE]: 1", "thought": True}, {"text": next(texts)},
        ]}}]}))

    monkeypatch.setattr(judge.urllib.request, "urlopen", urlopen)
    responses, score = judge.score_figure("pred-image", "gold-image")
    assert score == 50
    assert len(responses) == len(calls) == 3


def test_judge_retries_rate_limits_and_does_not_expose_key(judge, monkeypatch):
    attempts = []
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)

    def rate_limit(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.HTTPError("https://google.example", 429, "fake-google-judge-key", {}, None)

    monkeypatch.setattr(judge.urllib.request, "urlopen", rate_limit)
    with pytest.raises(RuntimeError, match="HTTP error 429") as error:
        judge._generate([])
    assert len(attempts) == 3
    assert "fake-google-judge-key" not in str(error.value)


def test_judge_reports_blocked_response_as_error(judge, monkeypatch):
    monkeypatch.setattr(judge.urllib.request, "urlopen", lambda *a, **k: io.StringIO('{}'))
    with pytest.raises(RuntimeError, match="no candidate"):
        judge._generate([])


def test_compute_launcher_consumes_key_before_submission(judge, monkeypatch):
    # The fixture imported the module exactly as the launcher does in a fresh
    # container, consuming FOH_JUDGE_API_KEY before compute_scores executes.
    monkeypatch.setitem(sys.modules, "gpt4_visual_judge", judge)
    calls = []

    def run_path(path, run_name):
        assert "FOH_JUDGE_API_KEY" not in os.environ
        assert path == "/testbed/foh_original_compute_scores.py"
        assert run_name == "__main__"
        calls.append(path)

    monkeypatch.setattr(driver.runpy, "run_path", run_path)
    exec(driver.COMPUTE_LAUNCHER, {})
    assert len(calls) == 1


def test_driver_mounts_readonly_adapter_and_keeps_key_out_of_build_env(monkeypatch, tmp_path):
    captured = []

    class Collection:
        def create(self, *args, **kwargs):
            captured.append(kwargs)
            return "container"

    original = Collection.create
    monkeypatch.setitem(sys.modules, "docker.models.containers", SimpleNamespace(ContainerCollection=Collection))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-judge-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "unrelated-azure-key")
    # main intentionally mutates the subprocess environment, argv and sys.path.
    monkeypatch.setattr(sys, "argv", ["driver", "--judge-model", "gemini-2.5-pro",
                                       "--judge-api-key-env", "GEMINI_API_KEY", "--", "--run_id", "test"])
    monkeypatch.setattr(sys, "path", list(sys.path))

    def run_module(name, run_name):
        assert name == "evaluation.harness.run_evaluation"
        assert sys.argv[1:] == ["--run_id", "test"]
        assert "GEMINI_API_KEY" not in os.environ
        assert os.environ["OPENAI_API_KEY"] == "foh-gemini-judge-no-openai-credential"
        assert "AZURE_OPENAI_KEY" not in os.environ
        assert Collection().create(image="test", volumes={
            str(tmp_path / "gpt4_visual_judge.py"): {"bind": "/testbed/gpt4_visual_judge.py"},
            str(tmp_path / "compute_scores.py"): {"bind": "/testbed/compute_scores.py"},
            "/dataset": {"bind": "/testbed/benchmark"},
        }) == "container"
        mounts = captured[0]["volumes"]
        assert mounts["/dataset"] == {"bind": "/testbed/benchmark"}
        destinations = {value["bind"]: Path(source) for source, value in mounts.items()}
        assert destinations["/testbed/foh_original_visual_judge.py"] == tmp_path / "gpt4_visual_judge.py"
        assert destinations["/testbed/foh_original_compute_scores.py"] == tmp_path / "compute_scores.py"
        assert destinations["/testbed/gpt4_visual_judge.py"].name == "scienceagentbench_visual_judge.py"
        for source, value in mounts.items():
            if value["bind"] != "/testbed/benchmark":
                assert value["mode"] == "ro"
        launcher = destinations["/testbed/compute_scores.py"].read_text()
        assert "fake-google-judge-key" not in launcher
        assert launcher == driver.COMPUTE_LAUNCHER
        environment = captured[0]["environment"]
        assert environment["FOH_JUDGE_API_KEY"] == "fake-google-judge-key"
        assert environment["FOH_JUDGE_MODEL"] == "gemini-2.5-pro"
        assert all(environment[name] == "" for name in driver.OPENAI_ENV_NAMES)

    monkeypatch.setattr(driver.runpy, "run_module", run_module)
    driver.main()
    assert Collection.create is original
    assert len(captured) == 1


def test_gemini_preflight_accepts_google_key_without_openai(monkeypatch):
    monkeypatch.setattr(sab, "HARNESS_MODULES", ())
    monkeypatch.setattr(sab.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(sab.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    sab.preflight(grading=True, environment={"GOOGLE_API_KEY": "test"}, judge_model="gemini-2.5-pro")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        sab.preflight(grading=True, environment={}, judge_model="gemini-2.5-pro")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        sab.preflight(grading=True, environment={"GOOGLE_API_KEY": "test"})


def test_custom_judge_command_and_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(sab, "purge_instance_images", lambda _: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-judge-key")

    def run(command, **kwargs):
        assert Path(command[2]).name == "scienceagentbench_eval_driver.py"
        assert command[3:8] == ["--judge-model", "gemini-2.5-pro", "--judge-api-key-env", "GEMINI_API_KEY", "--"]
        assert command[8] == "--benchmark_path"
        assert "fake-google-judge-key" not in str(command)
        result = tmp_path / "logs/run_evaluation/test/7/output/result.json"
        result.parent.mkdir(parents=True)
        result.write_text('[1, 1.0, 1, "passed"]')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sab.subprocess, "run", run)
    result = sab.run_official_eval(
        {"instance_id": 7}, bench_dir=tmp_path, benchmark_root=tmp_path / "benchmark",
        pred_dir=tmp_path / "preds", log_dir=tmp_path / "eval_logs", split="verified",
        run_id="test", cache_level="base", timeout=60, judge_model="gemini-2.5-pro",
    )
    assert result["score"] == 1.0
    assert sab.judge_metadata("gemini-2.5-pro") != sab.judge_metadata("gemini-2.5-flash")
    assert "fake-google-judge-key" not in json.dumps(sab.judge_metadata("gemini-2.5-pro"))


@pytest.mark.parametrize("args", [
    ["--judge-model", "gemini-2.5-pro", "--no-grade"],
    ["--judge-api-key-env", "MY_JUDGE_KEY"],
])
def test_conflicting_judge_flags_fail_before_benchmark_setup(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["science", *args])
    with pytest.raises(SystemExit) as error:
        sab.main()
    assert error.value.code == 2


@pytest.mark.parametrize("eval_only", [False, True])
def test_cli_passes_judge_to_scoring_and_checkpoint_identity(monkeypatch, tmp_path, eval_only):
    monkeypatch.setattr(sab, "resolve_checkout", lambda *a, **k: tmp_path)
    instances = [{"instance_id": i, "gold_program_name": f"task_{i}.py"} for i in (1, 2)]
    monkeypatch.setattr(sab, "load_instances", lambda *a, **k: instances)
    monkeypatch.setattr(sab, "validate_instances", lambda items, *a: items)
    monkeypatch.setattr(sab, "resolve_artifacts", lambda *a: tmp_path)
    monkeypatch.setattr(sab, "preflight", lambda **k: None)
    monkeypatch.setattr(sab, "preflight_external_hermes", lambda *a: None)
    monkeypatch.setattr(sab, "_resolve_eval_agent_version", lambda *a: "seed")
    monkeypatch.setattr(sab, "make_case", lambda instance, **k: SimpleNamespace(
        case_id=str(instance["instance_id"]), workspace=tmp_path))
    program = tmp_path / "program.py"
    program.write_text("print('result')")
    monkeypatch.setattr(sab, "locate_program", lambda *a: program)
    scored = []

    def score(instance, **kwargs):
        assert kwargs["judge_model"] == "gemini-2.5-pro"
        assert kwargs["judge_api_key"] == "explicit-test-key"
        scored.append(instance)
        return {"score": 1.0, "verifier_status": "completed"}

    monkeypatch.setattr(sab, "run_official_eval", score)

    def experiment(**kwargs):
        identity = kwargs["grade_identity"]
        assert identity["grader"] == "scienceagentbench_gemini_visual_judge"
        assert identity["judge"]["model"] == "gemini-2.5-pro"
        case = kwargs["cases"][0] if eval_only else kwargs["train_cases"][0]
        result = kwargs["grade"](case, SimpleNamespace(status="completed"))
        assert result["score"] == 1.0
        assert result["judge"] == identity["judge"]
        assert result["grading_type"] == identity["grader"]
        return {}

    monkeypatch.setattr(sab, "run_eval_only", experiment)
    monkeypatch.setattr(sab, "run_federated_experiment", experiment)
    args = ["science", "--judge-model", "google/gemini-2.5-pro", "--api_key",
            "explicit-test-key", "--output-dir", str(tmp_path / "run"), "--split", "0.5"]
    if eval_only:
        args.append("--eval-only")
    monkeypatch.setattr(sys, "argv", args)
    sab.main()
    assert len(scored) == 1
