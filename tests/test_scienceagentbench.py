"""No-provider tests for ScienceAgentBench selection, prompts, and grading."""
import json
from pathlib import Path

import pytest

from tasks.task_scienceagentbench import (
    ARTIFACT_DIRS,
    dataset_folder,
    locate_program,
    make_case,
    prepare_workspace,
    resolve_artifacts,
    run_official_eval,
    select_instances,
    split_instances,
    task_prompt,
    validate_instances,
)


def fake_instance(instance_id: int, *, domain="Bioinformatics", repo="owner/project", folder=None):
    folder = folder or f"data{instance_id}"
    return {
        "instance_id": instance_id,
        "domain": domain,
        "subtask_categories": "Data Analysis",
        "github_name": repo,
        "task_inst": f"Solve task {instance_id}.",
        "domain_knowledge": "Background notes.",
        "dataset_folder_tree": f"|-- {folder}/\n|---- input.csv",
        "dataset_preview": "[START Preview]\na,b\n[END Preview]",
        "src_file_or_path": "examples/demo.py",
        "gold_program_name": f"task_{instance_id}.py",
        "output_fname": f"pred_results/task_{instance_id}.csv",
        "eval_script_name": f"task_{instance_id}_eval.py",
    }


def test_selectors_cover_ids_domains_repos_and_prefixes():
    instances = [
        fake_instance(1, domain="Computational Chemistry", repo="deepchem/deepchem"),
        fake_instance(2, domain="Geographical Information Science", repo="rasterio/rasterio"),
        fake_instance(3, domain="Computational Chemistry", repo="owner/project"),
    ]

    assert select_instances(instances, "all") == instances
    assert select_instances(instances, "first_2") == instances[:2]
    assert select_instances(instances, "domain:computational-chemistry") == [
        instances[0],
        instances[2],
    ]
    assert select_instances(instances, "repo:rasterio") == [instances[1]]
    assert select_instances(instances, "2,task_3.py") == instances[1:]
    assert select_instances(instances, "nonexistent") == []


def test_split_is_reproducible_and_disjoint():
    instances = [fake_instance(index) for index in range(10)]
    train, held_out = split_instances(instances, 0.7, seed=42)
    train_again, held_out_again = split_instances(instances, 0.7, seed=42)

    assert len(train) == 7
    assert len(held_out) == 3
    assert train == train_again
    assert held_out == held_out_again
    assert not set(map(id, train)) & set(map(id, held_out))


def test_validate_rejects_structural_gaps_but_allows_empty_knowledge():
    instances = [fake_instance(1), fake_instance(2)]
    instances[0]["domain_knowledge"] = ""
    assert validate_instances(instances, "custom") == instances

    duplicated = [fake_instance(1), fake_instance(1)]
    with pytest.raises(ValueError, match="unique"):
        validate_instances(duplicated, "custom")

    broken = [fake_instance(1)]
    broken[0]["output_fname"] = ""
    with pytest.raises(ValueError, match="output_fname"):
        validate_instances(broken, "custom")


def test_dataset_folder_requires_the_upstream_tree_format():
    assert dataset_folder(fake_instance(1, folder="clintox")) == "clintox"

    malformed = fake_instance(2)
    malformed["dataset_folder_tree"] = "clintox/"
    with pytest.raises(ValueError, match="dataset tree"):
        dataset_folder(malformed)


def test_prompt_states_the_submission_contract():
    instance = fake_instance(1, folder="clintox")
    prompt = task_prompt(instance, use_knowledge=False)

    assert "pred_programs/pred_task_1.py" in prompt
    assert "benchmark/datasets/clintox/" in prompt
    assert "pred_results/task_1.csv" in prompt
    assert "Background notes." not in prompt
    assert "Background notes." in task_prompt(instance, use_knowledge=True)


def test_prepare_workspace_resets_to_declared_inputs(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "datasets" / "data1").mkdir(parents=True)
    (benchmark_root / "datasets" / "data1" / "input.csv").write_text("a,b\n")
    instance = fake_instance(1)
    case = make_case(
        instance,
        workspace_root=tmp_path / "workspaces",
        benchmark_root=benchmark_root,
        use_knowledge=False,
        copy_datasets=True,
    )
    case.workspace.mkdir(parents=True)
    (case.workspace / "stale.txt").write_text("from an earlier round")

    case.prepare(case.workspace)

    assert not (case.workspace / "stale.txt").exists()
    assert (case.workspace / "pred_programs").is_dir()
    assert (case.workspace / "pred_results").is_dir()
    assert (case.workspace / "benchmark" / "datasets" / "data1" / "input.csv").is_file()
    assert case.metadata["program_path"] == "pred_programs/pred_task_1.py"


def test_prepare_workspace_refuses_a_foreign_directory(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "datasets" / "data1").mkdir(parents=True)
    with pytest.raises(ValueError, match="Refusing to reset"):
        prepare_workspace(
            tmp_path / "not-the-instance-id",
            fake_instance(1),
            benchmark_root,
            copy_datasets=True,
        )


def test_locate_program_prefers_the_requested_path(tmp_path):
    instance = fake_instance(1)
    assert locate_program(tmp_path, instance) is None

    (tmp_path / "pred_programs").mkdir()
    (tmp_path / "pred_programs" / "task_1.py").write_text("print(1)")
    assert locate_program(tmp_path, instance).name == "task_1.py"

    (tmp_path / "pred_programs" / "pred_task_1.py").write_text("print(1)")
    assert locate_program(tmp_path, instance).name == "pred_task_1.py"


def test_missing_artifacts_point_at_the_release_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark_verified.zip"):
        resolve_artifacts(tmp_path)

    for name in ARTIFACT_DIRS:
        (tmp_path / "benchmark" / name).mkdir(parents=True)
    assert resolve_artifacts(tmp_path) == tmp_path / "benchmark"


def stub_harness(bench_dir: Path, body: str) -> None:
    harness = bench_dir / "evaluation" / "harness"
    harness.mkdir(parents=True)
    (bench_dir / "evaluation" / "__init__.py").write_text("")
    (harness / "__init__.py").write_text("")
    (harness / "run_evaluation.py").write_text(body)


HARNESS_TEMPLATE = '''
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
for flag in (
    "--benchmark_path", "--pred_program_path", "--log_fname", "--dataset_name",
    "--split", "--cache_level", "--clean", "--run_id", "--max_workers", "--timeout",
):
    parser.add_argument(flag)
parser.add_argument("--instance_ids", nargs="+")
args = parser.parse_args()
output = Path("logs/run_evaluation") / args.run_id / args.instance_ids[0] / "output"
output.mkdir(parents=True, exist_ok=True)
(output / "result.json").write_text(json.dumps({payload}))
'''


def test_official_result_is_reported_verbatim(tmp_path):
    stub_harness(tmp_path, HARNESS_TEMPLATE.format(payload="[1, 0.42, 0, 'values differ']"))
    result = run_official_eval(
        fake_instance(7),
        bench_dir=tmp_path,
        benchmark_root=tmp_path / "benchmark",
        pred_dir=tmp_path / "preds",
        log_dir=tmp_path / "eval_logs",
        split="verified",
        run_id="foh-test-0001-7",
        cache_level="base",
        timeout=60,
    )

    assert result["verifier_status"] == "completed"
    assert result["score"] == 0.0
    assert result["valid_program"] == 1
    assert result["codebert_score"] == pytest.approx(0.42)
    assert result["log_info"] == "values differ"
    assert (tmp_path / "eval_logs" / "foh-test-0001-7.log").is_file()

    stub_harness_dir = tmp_path / "evaluation" / "harness" / "run_evaluation.py"
    stub_harness_dir.write_text(HARNESS_TEMPLATE.format(payload="[1, 0.9, 1, 'ok']"))
    passed = run_official_eval(
        fake_instance(7),
        bench_dir=tmp_path,
        benchmark_root=tmp_path / "benchmark",
        pred_dir=tmp_path / "preds",
        log_dir=tmp_path / "eval_logs",
        split="verified",
        run_id="foh-test-0002-7",
        cache_level="base",
        timeout=60,
    )
    assert passed["score"] == 1.0
    assert passed["success_rate"] == 1


def test_a_harness_crash_scores_zero_without_raising(tmp_path):
    stub_harness(tmp_path, 'import sys\nprint("build failed", file=sys.stderr)\nsys.exit(3)\n')
    result = run_official_eval(
        fake_instance(7),
        bench_dir=tmp_path,
        benchmark_root=tmp_path / "benchmark",
        pred_dir=tmp_path / "preds",
        log_dir=tmp_path / "eval_logs",
        split="verified",
        run_id="foh-test-0003-7",
        cache_level="base",
        timeout=60,
    )

    assert result["verifier_status"] == "error"
    assert result["score"] == 0.0
    assert "build failed" in result["harness_output"]
    assert json.loads(json.dumps(result))  # payloads stay JSON-serialisable
