#!/bin/bash
# MMLU, chain-of-thought, temperature 0.
#
# gpt-oss is a reasoning model: with a small max_tokens the whole budget is
# spent in reasoning_content, "content" comes back null and finish_reason is
# "length", so every answer is unparsable and the score is 0. Asked with room
# to reason it scores 0.8247. Note this is not directly comparable to entries
# that answered with a bare letter under a small token cap.
COT=1 python qa_eval.py mmlu_all_questions.json mmlu mmlu.jsonl openai/gpt-oss-20b
