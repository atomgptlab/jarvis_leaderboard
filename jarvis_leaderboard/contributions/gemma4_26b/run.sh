#!/bin/bash
# 0-shot MMLU against the AtomGPT endpoint (gemma-4-26b), temperature 0.
#
# Question text: the figshare copy used by earlier contributions
# (https://figshare.com/ndownloader/files/44211497) now returns an empty 202,
# so questions come from HuggingFace cais/mmlu and are paired to this repo's
# ids per subject, in order. The pairing is accepted only if all 14042
# reconstructed answers equal the stored ground truth, which they do.
#
# Each question is asked on its own, with the four options labelled A-D and
# the model asked for a single letter. A reply with no parsable letter is
# recorded as wrong rather than dropped, so the denominator stays 14042.
python mmlu_build.py mmlu_test_quiz.json mmlu_questions.json
python mmlu_eval.py mmlu_questions.json mmlu_gemma4.jsonl gemma-4-26b
