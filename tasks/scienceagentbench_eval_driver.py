"""Run pinned ScienceAgentBench with a per-container Gemini visual judge.

No upstream files are edited. Only the judge and a credential-consuming
compute_scores launcher are mounted over their container paths.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import tempfile


OPENAI_ENV_NAMES = (
    "OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME",
)
COMPUTE_LAUNCHER = '''import gpt4_visual_judge  # consume the key before running submitted code
import runpy
runpy.run_path("/testbed/foh_original_compute_scores.py", run_name="__main__")
'''


def install_judge_mounts(collection_type, *, bench_dir, launcher, model, api_key):
    """Intercept the pinned harness's container creation inside this process."""
    original_create = collection_type.create
    adapter = Path(__file__).with_name("scienceagentbench_visual_judge.py").resolve()

    def create(collection, *args, **kwargs):
        volumes = dict(kwargs.get("volumes") or {})
        expected = {"/testbed/gpt4_visual_judge.py", "/testbed/compute_scores.py"}
        found = {value.get("bind") for value in volumes.values() if isinstance(value, dict)}
        if not expected.issubset(found):
            raise RuntimeError("Pinned harness container mounts changed; cannot install Gemini judge")
        volumes = {source: value for source, value in volumes.items()
                   if value.get("bind") not in expected}
        for source, target in (
            (adapter, "/testbed/gpt4_visual_judge.py"),
            (launcher, "/testbed/compute_scores.py"),
            (bench_dir / "gpt4_visual_judge.py", "/testbed/foh_original_visual_judge.py"),
            (bench_dir / "compute_scores.py", "/testbed/foh_original_compute_scores.py"),
        ):
            volumes[str(source)] = {"bind": target, "mode": "ro"}
        environment = dict(kwargs.get("environment") or {})
        environment.update({name: "" for name in OPENAI_ENV_NAMES})
        environment.update(FOH_JUDGE_MODEL=model, FOH_JUDGE_API_KEY=api_key)
        return original_create(collection, *args, **{
            **kwargs, "volumes": volumes, "environment": environment,
        })

    collection_type.create = create
    return original_create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-api-key-env", required=True)
    args, upstream_args = parser.parse_known_args()
    if upstream_args[:1] == ["--"]:
        upstream_args = upstream_args[1:]
    api_key = os.environ.pop(args.judge_api_key_env, "")
    if not api_key:
        parser.error(f"Missing Gemini judge credential: set {args.judge_api_key_env}")
    # The pinned CLI insists on an OpenAI credential and embeds those variables
    # in Dockerfiles. Supply a non-secret marker, never the Google key.
    for name in OPENAI_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ["OPENAI_API_KEY"] = "foh-gemini-judge-no-openai-credential"
    bench_dir = Path.cwd().resolve()
    sys.path.insert(0, str(bench_dir))
    from docker.models.containers import ContainerCollection

    with tempfile.TemporaryDirectory(prefix="foh-sab-judge-") as temporary:
        launcher = Path(temporary) / "compute_scores.py"
        launcher.write_text(COMPUTE_LAUNCHER, encoding="utf-8")
        original = install_judge_mounts(
            ContainerCollection, bench_dir=bench_dir, launcher=launcher,
            model=args.judge_model, api_key=api_key,
        )
        try:
            sys.argv = ["evaluation.harness.run_evaluation", *upstream_args]
            runpy.run_module("evaluation.harness.run_evaluation", run_name="__main__")
        finally:
            ContainerCollection.create = original


if __name__ == "__main__":
    main()
