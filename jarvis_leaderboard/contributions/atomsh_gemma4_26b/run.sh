#!/bin/bash
# Terminal-Bench core 0.1.1, 80 tasks, one attempt each, 4 concurrent.
#
# tb_agents/atomsh_agent.py is a Terminal-Bench "installed agent": the harness
# installs atomsh into every task container and runs one non-interactive
# prompt carrying the task instruction. A task counts as resolved only if
# every pytest assertion in it passes, so there is no partial credit.
#
# source=pypi installs the published atomsh; source=local embeds a wheel built
# from a working tree as base64 in the setup script, for measuring an
# unreleased change. Set ATOMSH_API_KEY, or log in first with `atomsh login`.
#
# One detail that is easy to lose: the setup script installs with
# --ignore-installed, because some task images carry distro-managed copies of
# atomsh's dependencies with no RECORD file, and pip aborts the whole install
# rather than replace one. Without it the agent is silently absent and the
# task scores zero having made no tool calls at all.
pip install terminal-bench==0.2.18

PYTHONPATH=. tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path tb_agents.atomsh_agent:AtomshAgent \
  -k source=pypi \
  --n-concurrent 4 \
  --n-attempts 1 \
  --output-path runs/atomsh
