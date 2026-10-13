"""Apptainer verifier backend: Dockerfile translation and grade mapping, no containers."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks import scienceagentbench_apptainer as app
from tasks import task_scienceagentbench as sab


def test_parser_follows_docker_continuation_and_comment_rules():
    steps = app.parse_dockerfile(
        "FROM --platform=linux/x86_64 ubuntu:22.04\n"
        "# top comment\n"
        "RUN set -e; \\\n"
        "    echo one; \\\n"
        "    \n"
        "    # inner comment\n"
        "    echo two\n"
        "ENV PATH=/opt/x/bin:$PATH\n"
    )
    assert steps == [
        ("FROM", "--platform=linux/x86_64 ubuntu:22.04"),
        ("RUN", "set -e;     echo one;     echo two"),
        ("ENV", "PATH=/opt/x/bin:$PATH"),
    ]


def test_base_definition_keeps_upstream_order_and_adds_mount_points():
    definition = app.base_definition(
        "FROM --platform=linux/x86_64 ubuntu:22.04\nARG DEBIAN_FRONTEND=noninteractive\n"
        "RUN echo 'a b'\nENV PATH=/opt/miniconda3/bin:$PATH\nRUN conda init\n"
    )
    assert definition.startswith("Bootstrap: docker\nFrom: ubuntu:22.04\n")
    post = definition.split("%post", 1)[1]
    assert post.index("echo") < post.index("export PATH") < post.index("conda init")
    assert "export PATH=/opt/miniconda3/bin:$PATH" in definition.split("%post")[0]
    assert "/testbed/instance_path" in post and "touch /testbed/compute_scores.py" in post


def test_instance_script_reads_copy_sources_from_build_context():
    script = app.instance_script(
        "FROM base\nRUN mkdir -p /testbed/p/\nCOPY ./pred_x.py /testbed/p/pred_x.py\n"
        "WORKDIR /testbed/\nENV OPENAI_API_KEY=secret\nRUN pip install -r r.txt\n"
    )
    assert "cp /foh_build/pred_x.py /testbed/p/pred_x.py" in script
    assert "cd /testbed/" in script
    assert "secret" not in script
    assert script.splitlines()[0] == "set -e"


def test_translation_accepts_the_pinned_upstream_dockerfiles():
    bench = Path(sab.DEFAULT_BENCH_DIR)
    if not (bench / "evaluation" / "harness" / "dockerfiles.py").is_file():
        pytest.skip("pinned checkout not present")
    dockerfiles, _ = app._upstream(bench)
    assert "From: ubuntu:22.04" in app.base_definition(
        dockerfiles.get_dockerfile_base("linux/x86_64", "x86_64"))
    assert "pipreqs" in app.instance_script(
        dockerfiles.get_dockerfile_instance("linux/x86_64", "base", "pred_t.py"))


def test_apptainer_grade_maps_the_official_tuple(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "run_eval", lambda *a, **k: ((1, 0.8, 1, "ok"), {"run_seconds": 3}))
    result = sab.run_apptainer_eval(
        {"instance_id": "7"}, bench_dir=tmp_path, benchmark_root=tmp_path, pred_dir=tmp_path,
        log_dir=tmp_path, run_id="r", timeout=10, base_image=tmp_path / "b.sif", scratch=tmp_path,
    )
    assert result["score"] == 1.0 and result["container_runtime"] == "apptainer"
    assert result["run_seconds"] == 3


def test_apptainer_crash_or_missing_result_scores_zero(monkeypatch, tmp_path):
    common = dict(bench_dir=tmp_path, benchmark_root=tmp_path, pred_dir=tmp_path, log_dir=tmp_path,
                  run_id="r", timeout=10, base_image=tmp_path / "b.sif", scratch=tmp_path)
    monkeypatch.setattr(app, "run_eval", lambda *a, **k: (None, {"error": "build failed"}))
    assert sab.run_apptainer_eval({"instance_id": "7"}, **common)["score"] == 0.0

    def boom(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(app, "run_eval", boom)
    result = sab.run_apptainer_eval({"instance_id": "7"}, **common)
    assert result["score"] == 0.0 and result["verifier_status"] == "error"


def test_apptainer_preflight_needs_no_docker(monkeypatch):
    monkeypatch.setattr(sab.shutil, "which", lambda name: "/usr/bin/apptainer" if name == "apptainer" else None)
    sab.preflight(grading=True, environment={"GEMINI_API_KEY": "k"}, judge_model="gemini-2.5-pro",
                  container_runtime="apptainer")
    with pytest.raises(RuntimeError, match="Google API key"):
        sab.preflight(grading=True, environment={}, judge_model="gemini-2.5-pro",
                      container_runtime="apptainer")


def test_cli_routes_grades_through_apptainer(monkeypatch, tmp_path):
    monkeypatch.setattr(sab, "resolve_checkout", lambda *a, **k: tmp_path)
    instances = [{"instance_id": i, "gold_program_name": f"task_{i}.py"} for i in (1, 2)]
    monkeypatch.setattr(sab, "load_instances", lambda *a, **k: instances)
    monkeypatch.setattr(sab, "validate_instances", lambda items, *a: items)
    monkeypatch.setattr(sab, "resolve_artifacts", lambda *a: tmp_path)
    monkeypatch.setattr(sab, "preflight", lambda **k: None)
    monkeypatch.setattr(sab, "preflight_external_hermes", lambda *a: None)
    monkeypatch.setattr(app, "ensure_base_image", lambda *a, **k: tmp_path / "base.sif")
    monkeypatch.setattr(sab, "make_case", lambda instance, **k: SimpleNamespace(
        case_id=str(instance["instance_id"]), workspace=tmp_path))
    program = tmp_path / "program.py"
    program.write_text("print('result')")
    monkeypatch.setattr(sab, "locate_program", lambda *a: program)
    monkeypatch.setattr(sab, "run_official_eval", lambda *a, **k: pytest.fail("docker path used"))
    seen = {}

    def apptainer_eval(instance, **kwargs):
        seen.update(kwargs)
        return {"score": 1.0, "verifier_status": "completed"}

    monkeypatch.setattr(sab, "run_apptainer_eval", apptainer_eval)

    def experiment(**kwargs):
        assert kwargs["grade_identity"]["container_runtime"] == "apptainer"
        assert kwargs["num_workers"] == 16
        case = kwargs["train_cases"][0]
        assert kwargs["grade"](case, SimpleNamespace(status="completed"))["score"] == 1.0
        return {}

    monkeypatch.setattr(sab, "run_federated_experiment", experiment)
    monkeypatch.setattr(sys, "argv", [
        "science", "--api_key", "fake", "--container-runtime", "apptainer", "--num-workers", "16",
        "--apptainer-scratch", str(tmp_path / "scratch"), "--judge-model", "gemini-2.5-pro",
        "--output-dir", str(tmp_path / "run"), "--split", "0.5"])
    sab.main()
    assert seen["base_image"] == tmp_path / "base.sif"
    assert seen["judge_model"] == "gemini-2.5-pro"


def test_local_recipe_reads_the_pinned_base_dockerfile():
    recipe = sab.sab_local.base_recipe(
        "FROM ubuntu:22.04\nRUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-x.sh' -O m.sh\n"
        "RUN conda init --all && conda config --append channels conda-forge\n"
        "RUN conda init && conda install python=3.10 pip -y\n"
        "RUN /opt/miniconda3/bin/pip install \"numpy<2.0\" pipreqs\n"
    )
    assert recipe["installer"].endswith("Miniconda3-x.sh")
    assert recipe["conda_install_args"] == "python=3.10 pip -y"
    assert recipe["append_conda_forge"] is True
    assert "numpy<2.0" in recipe["pip_install"]


def test_cli_routes_grades_through_local_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(sab, "resolve_checkout", lambda *a, **k: tmp_path)
    instances = [{"instance_id": i, "gold_program_name": f"task_{i}.py"} for i in (1, 2)]
    monkeypatch.setattr(sab, "load_instances", lambda *a, **k: instances)
    monkeypatch.setattr(sab, "validate_instances", lambda items, *a: items)
    monkeypatch.setattr(sab, "resolve_artifacts", lambda *a: tmp_path)
    monkeypatch.setattr(sab, "preflight", lambda **k: None)
    monkeypatch.setattr(sab, "preflight_external_hermes", lambda *a: None)
    monkeypatch.setattr(sab.sab_local, "ensure_base_env", lambda *a, **k: tmp_path / "base")
    monkeypatch.setattr(sab, "make_case", lambda instance, **k: SimpleNamespace(
        case_id=str(instance["instance_id"]), workspace=tmp_path))
    program = tmp_path / "program.py"
    program.write_text("print('result')")
    monkeypatch.setattr(sab, "locate_program", lambda *a: program)
    monkeypatch.setattr(sab, "run_official_eval", lambda *a, **k: pytest.fail("docker path used"))
    seen = {}
    monkeypatch.setattr(sab.sab_local, "run_eval",
                        lambda instance, **k: (seen.update(k) or (1, 0.5, 1, "ok"), {}))

    def experiment(**kwargs):
        assert kwargs["grade_identity"]["container_runtime"] == "local"
        case = kwargs["train_cases"][0]
        grade = kwargs["grade"](case, SimpleNamespace(status="completed"))
        assert grade["score"] == 1.0 and grade["container_runtime"] == "local"
        return {}

    monkeypatch.setattr(sab, "run_federated_experiment", experiment)
    monkeypatch.setenv("GEMINI_API_KEY", "wrong-environment-key")
    monkeypatch.setattr(sys, "argv", [
        "science", "--api_key", "fake", "--container-runtime", "local", "--num-workers", "16",
        "--apptainer-scratch", str(tmp_path / "scratch"), "--judge-model", "gemini-2.5-pro",
        "--output-dir", str(tmp_path / "run"), "--split", "0.5"])
    sab.main()
    assert seen["base_prefix"] == tmp_path / "base"
    assert seen["judge_api_key"] == "fake"
    assert "base_image" not in seen
