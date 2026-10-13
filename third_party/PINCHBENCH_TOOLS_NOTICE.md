# PinchBench integration toolchain notice

`scripts/bootstrap_pinchbench_tools.py` downloads a local, ignored runtime
under `.pinchbench-tools/`. The repository does not vendor these binaries.

- Node.js 22.23.2 — Node.js contributors; see the upstream Node.js license.
- Google Workspace CLI 0.22.5 — Apache License 2.0; this project states that
  it is not an officially supported Google product.
- GitHub CLI 2.98.0 — MIT License, GitHub, Inc.
- `@juppytt/fws` 0.4.4 — source revision
  `d9f7e3067769814f28ce5fd05888f586bd52977e`; the published npm metadata and
  source tag do not declare a license. Review its upstream terms before
  redistributing the downloaded package.

Authoritative upstream locations and archive checksums are recorded in
`pinchbench-tools.lock.json`. The npm transitive dependency graph is fixed by
`pinchbench-fws-package-lock.json`.
