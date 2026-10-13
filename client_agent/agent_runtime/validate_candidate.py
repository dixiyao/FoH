"""Protected command-line validation for an evolvable role overlay.

External Hermes owns its upstream tool registry and therefore cannot see the
in-process FoH ``run_validation`` callback.  This module exposes the same
fixed validator as a normal command inside the task sandbox.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .evolution import validate_agent_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument(
        "--role", required=True, choices=("client_agent", "server_agent")
    )
    args = parser.parse_args()
    result = validate_agent_directory(args.directory, target_role=args.role)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
