#!/bin/bash
# GPQA-diamond and AIME 2024, chain-of-thought, temperature 0.
#
# Protocol matters more than the model here: asking gemma-4-26b for a bare
# letter with a 12-token cap scores 0.3939 on GPQA, while letting it reason
# scores 0.6667 - 27 points from the same model on the same questions. These
# entries are the reasoning form, which is how both benchmarks are normally
# reported. A reply with no parsable answer counts as wrong, not dropped.
#
# GPQA ships the correct answer and three distractors in separate columns, so
# the A-D order is assigned here by shuffling with seed 42. The seed is part
# of the ground truth: a different seed gives a different answer key.
python build_gpqa_aime.py benchmarks/AI/TextClass questions.json
COT=1 python qa_eval.py questions.json gpqa_diamond gpqa.jsonl gemma-4-26b
python qa_eval.py questions.json aime_2024 aime.jsonl gemma-4-26b
