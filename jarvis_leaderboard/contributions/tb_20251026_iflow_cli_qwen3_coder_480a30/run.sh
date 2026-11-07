#!/bin/bash
# This entry is NOT our measurement. It is iflow-cli + Qwen3-Coder-480A30's own
# Terminal-Bench run, recomputed here from the per-task results.json files the
# submitting team published as a condition of submitting to the official
# leaderboard:
#
#   https://github.com/laude-institute/terminal-bench-leaderboard/tree/main/results/terminal-bench-core%400.1.1/20251026_iflow-cli_Qwen3-Coder-480A30
#
# Reproduce it by fetching that directory and counting resolved tasks in the
# FIRST trial - not the five-trial mean - so that every entry on this
# benchmark is one attempt per task, matching the entries we ran ourselves.
# The five-trial mean is recorded in metadata.json.
set -euo pipefail

git clone --depth 1 https://github.com/laude-institute/terminal-bench-leaderboard
cd terminal-bench-leaderboard/results/terminal-bench-core@0.1.1/20251026_iflow-cli_Qwen3-Coder-480A30

python - <<'EOF'
import glob, json, os
trials = sorted(d for d in glob.glob("*") if os.path.isdir(d))
res = json.load(open(os.path.join(trials[0], "results.json")))["results"]
ok = sum(1 for r in res if r.get("is_resolved"))
print(f"{ok}/{len(res)} = {ok/len(res):.4f}")
for r in res:
    print(r["task_id"], 1 if r.get("is_resolved") else 0)
EOF
