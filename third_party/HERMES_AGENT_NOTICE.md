# Hermes Agent upstream notice

Federation-over-Harness integrates the upstream **Hermes Agent** project by
Nous Research through an isolated process bridge. The bootstrap script checks
out the exact release recorded in `hermes-agent.lock.json`; it does not track a
moving branch and does not duplicate the source tree into each client/server
version.

- Upstream: <https://github.com/NousResearch/hermes-agent>
- Release: `v2026.8.19` (`0.20.5`)
- Commit: `fcbd1076a93841fa88855acce810e342a5b78101`
- License: MIT

The installed checkout retains upstream's `LICENSE` and all nested notices.
Hermes Agent's name and identity text remain attributed to Nous Research.
