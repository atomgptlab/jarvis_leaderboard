# ScoreHour

How accurate is an open-weight reasoning model, and **how much accuracy does
it deliver per hour** on hardware you can put under a desk? ScoreHour reports
both for GPQA Diamond. It covers 198 graduate-level multiple-choice questions
in biology, chemistry and physics, written so that the answers are hard to
look up.

Every model is served locally behind an OpenAI-compatible endpoint (vLLM
unless noted) on one of two machines:

- **RTX PRO 6000 Blackwell**: a workstation GPU with 96 GB of GDDR7 at about
  1.8 TB/s.
- **GB10**: a Grace Blackwell superchip (DGX Spark class) with 128 GB of
  unified LPDDR5x memory at about 273 GB/s.

Where a model ran on both machines, the same checkpoint, chat template,
sampling parameters, token budget and seeds were used, so the two rows differ
only in the hardware.

## Protocol

- 198 questions × **2 samples** each, single user turn with the
  [simple-evals](https://github.com/openai/simple-evals) prompt, no system
  prompt, no tools.
- **32,768-token budget** for reasoning and answer together. A response that
  reaches the budget without an answer is scored wrong, not rescued.
- **Accuracy** is the mean over all 396 question-sample pairs (± std over the
  two sample runs).
- **Time** is the wall-clock time during which at least one request was in
  flight, with up to *c* concurrent requests.
- **Score/h** = accuracy (%) / time (h). It is the accuracy delivered per hour
  of serving on the stated hardware. It is not given for models below the 25%
  chance level, which would otherwise rank first by answering fast.
- Only runs that answered all 396 requests are listed.
- &dagger; marks runs where the GPU was shared with a production server.

The two rankings disagree. The most accurate runs take hours on a GB10,
while small models that answer quickly lead on Score/h. Which one matters
depends on whether you need the best answer or many good-enough answers, for
example inside an agent loop.

## Relation to the GPQA Diamond leaderboard entry

Each row is also a regular contribution to the
[AI / TextClass / gpqa_diamond](https://atomgptlab.github.io/jarvis_leaderboard/AI/TextClass/gpqa_diamond_test_quiz)
benchmark. That benchmark takes one answer per question, so the contribution
holds **sample 0** and its leaderboard accuracy differs slightly from the
two-sample mean shown here. The leaderboard shuffles the answer options with
seed 42 and the harness uses its own order. Every predicted letter is mapped
through the option text it names, and the mapping is checked against the
leaderboard's answer key. The two-sample numbers, run time and Score/h are
stored in each entry's `metadata.json` under `"scorehour"`.

## Reproducing a row

```bash
git clone https://github.com/atomgptlab/reprollm
cd reprollm && pip install -e .
# serve the model as configured in configs/models.yaml, then:
python -m reprollm.cli run --models gemma-4-e2b --benchmarks gpqa_diamond --n 2
JARVIS_AUTHOR_EMAIL=you@example.org \
    python scripts/export_jarvis.py /path/to/jarvis_leaderboard
```

Every response (reasoning, answer, token counts, latency) is kept in the
[reprollm results](https://github.com/atomgptlab/reprollm/tree/main/results),
and every contribution's `run.sh` names the configuration it used.

## Accuracy against run time

Hover over a point for the model and its Score/h. Points toward the upper
left are both accurate and fast.

<!--table_content-->

## All runs, sorted by Score/h

Click a column header to sort.

<!--table_details-->

Reference: GPQA, [Rein et al., 2023](https://arxiv.org/abs/2311.12022) ·
[AGAPI agents](https://doi.org/10.1021/acs.jpclett.6c00837) ·
[AtomGPT](https://doi.org/10.1021/acs.jpclett.4c01126)
