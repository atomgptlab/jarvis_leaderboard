"""Pair jarvis_leaderboard's MMLU ids with their question text.

The figshare copy the original contribution used now returns an empty 202, so
the questions come from HuggingFace. The ids carry a global running index whose
ordering is not the HuggingFace ordering, so instead of reproducing that order
the two are matched per subject, in order. Nothing is assumed: the pairing is
accepted only if all 14042 reconstructed answers equal the stored ground truth.
"""
import json, sys, collections
from datasets import load_dataset

gt = json.load(open(sys.argv[1]))["test"]
out_path = sys.argv[2]
LETTERS = "ABCD"

# ids per subject, in file order
by_subject = collections.OrderedDict()
for key in gt:
    subject = key.rsplit("-", 1)[0].split("-")[-1]
    by_subject.setdefault(subject, []).append(key)

ds = load_dataset("cais/mmlu", "all", split="test")
hf = collections.OrderedDict()
for row in ds:
    hf.setdefault(row["subject"], []).append(row)

print(f"subjects: gt={len(by_subject)} hf={len(hf)}")
mismatched = [s for s in by_subject
              if len(by_subject[s]) != len(hf.get(s, []))]
if mismatched:
    sys.exit(f"per-subject counts differ: {mismatched[:5]}")

built, bad = {}, []
for subject, keys in by_subject.items():
    for key, row in zip(keys, hf[subject]):
        answer = LETTERS[row["answer"]]
        if answer != gt[key]:
            bad.append(key)
        built[key] = {"question": row["question"],
                      "choices": list(row["choices"]),
                      "answer": answer}

print(f"paired: {len(built)} | answer mismatches: {len(bad)}")
if bad:
    print("  sample:", bad[:3])
    sys.exit("pairing does not reproduce the ground truth answers")

json.dump(built, open(out_path, "w"))
print("VERIFIED against ground truth — wrote", out_path)
