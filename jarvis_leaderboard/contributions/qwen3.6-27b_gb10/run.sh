#!/bin/bash
# GPQA Diamond, ScoreHour protocol (198 questions x 2 samples, 32,768-token budget,
# simple-evals prompt). Serve the model behind an OpenAI-compatible endpoint as
# configured for 'qwen3.6-27b-gb10' in reprollm/configs/models.yaml, then run the benchmark
# and export sample 0 to this folder (the option order is mapped to the
# leaderboard's seed-42 answer key).
git clone https://github.com/atomgptlab/reprollm
cd reprollm && pip install -e .
python -m reprollm.cli run --models qwen3.6-27b-gb10 --benchmarks gpqa_diamond --n 2
JARVIS_AUTHOR_EMAIL=you@example.org \
    python scripts/export_jarvis.py /path/to/jarvis_leaderboard
