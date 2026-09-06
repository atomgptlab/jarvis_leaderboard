#!/bin/bash
# Terminal-Bench core 0.1.1, 80 tasks, one attempt each.
#
# Atomsh is driven through a Terminal-Bench "installed agent" adapter
# (tb_agents/atomsh_agent.py): the harness installs atomsh into each task
# container and runs one non-interactive prompt with the task instruction.
# A task counts as resolved only if every pytest assertion in it passes.
pip install terminal-bench==0.2.18

tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path tb_agents.atomsh_agent:AtomshAgent \
  -k source=pypi \
  --n-concurrent 4 \
  --n-attempts 1 \
  --output-path runs/atomsh
