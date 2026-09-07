# AgentBench

Terminal agents on Terminal-Bench core 0.1.1: 80 hand-written tasks covering
compilation, server setup, system administration, data science and security.
Each task hands the agent an instruction and a Docker container, and is graded
by pytest assertions run against the final container state. A task counts as
resolved only if **every** assertion passes, so there is no partial credit, and
accuracy is simply the fraction of the 80 tasks resolved.

Every entry here is **one attempt per task**. The third-party submissions each
ran five trials; the first trial is used so that all entries are measured the
same way, and the five-trial mean is recorded in each entry's `metadata.json`
(it moves the numbers by at most about two points).

Two things worth knowing before reading the table. No published agent solves
more than about two thirds of these tasks, so the practical ceiling is ~0.64
rather than 1.0. And some tasks penalise an agent for reasons unrelated to
skill - one grader could not decode the video codec an agent legitimately
chose, failing a task whose output was otherwise correct.

Every task in every entry was actually attempted. Nine tasks in the first
atomsh run were refused by an account rate limit before the agent made a
single tool call; those nine were rerun once the limit was lifted and three of
them resolve, which is the difference between 0.25 and the 0.2875 shown here.

## Reproducing these numbers

The atomsh entry, end to end:

```bash
pip install terminal-bench==0.2.18
git clone https://github.com/atomgptlab/jarvis_leaderboard
cd jarvis_leaderboard/jarvis_leaderboard/contributions/atomsh_gemma4_26b

export ATOMSH_API_KEY=...        # or: atomsh login
PYTHONPATH=. tb run \
  --dataset terminal-bench-core==0.1.1 \
  --agent-import-path tb_agents.atomsh_agent:AtomshAgent \
  -k source=pypi --n-concurrent 4 --n-attempts 1 \
  --output-path runs/atomsh
```

`tb_agents/atomsh_agent.py` in that directory is the whole adapter: it installs
the agent into each task container and runs one non-interactive prompt.
Swapping `--agent-import-path` for `-a claude-code`, `-a goose` or any other
harness reproduces the comparison rows the same way.

## Where the comparison data comes from

The third-party rows are not our measurements. They are recomputed from the
per-task `results.json` files that each team published as a condition of
submitting to the official leaderboard:

<https://github.com/laude-institute/terminal-bench-leaderboard/tree/main/results/terminal-bench-core%400.1.1>

Each entry's `metadata.json` links to its own source directory there, so any
row can be traced back to the trajectories it came from. Only the scores are
reproduced here; the trajectories stay in the upstream repository.

Reference: [Terminal-Bench](https://www.tbench.ai) ·
[harness](https://github.com/laude-institute/terminal-bench) ·
[run logs](https://github.com/laude-institute/terminal-bench-leaderboard)

<!--benchmark_description-->

<!--table_content-->
