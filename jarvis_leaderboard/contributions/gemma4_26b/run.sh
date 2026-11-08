#!/bin/bash
# 0-shot MMLU against the AtomGPT endpoint, temperature 0, bare-letter answers.
#
# Protocol note, because it dominates the score: this asks for a single letter
# with a 12-token cap and no room to reason, matching the older entries on
# this benchmark. Letting the same model reason is worth about 3.6 points on
# MMLU (measured on a 500-question subset: 0.846 -> 0.882), and far more on a
# reasoning benchmark. Compare only against entries using the same protocol.
#
# mmlu_build.py rebuilds the questions: the figshare copy earlier
# contributions used (ndownloader/files/44211497) now returns an empty 202, so
# they come from HuggingFace cais/mmlu and are paired to this repo's ids per
# subject. The pairing is accepted only if all 14042 reconstructed answers
# equal the stored ground truth.
#
# An unparsable reply counts as wrong rather than being dropped, so the
# denominator stays 14042 and the score is not flattered by silence.
python mmlu_build.py mmlu_test_quiz.json mmlu_questions.json
python mmlu_eval.py mmlu_questions.json mmlu.jsonl gemma-4-26b
