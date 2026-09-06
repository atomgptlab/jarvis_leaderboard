"""Build jarvis_leaderboard ground truth for GPQA-diamond and AIME 2024.

GPQA ships the correct answer and three distractors in separate columns, so
the A-D assignment has to be made here. It is shuffled with a fixed seed and
the seed recorded, because an unreproducible option order would make the
ground truth impossible for anyone else to regenerate.
"""
import json, os, random, sys, zipfile
from datasets import load_dataset

OUT_BENCH, OUT_Q = sys.argv[1], sys.argv[2]
SEED = 42
LETTERS = "ABCD"


def gpqa():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(SEED)
    gt, questions = {}, {}
    for i, row in enumerate(ds):
        options = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        options = [str(o).strip() for o in options]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[j] for j in order]
        key = f"gpqa_diamond-{i + 1}"
        gt[key] = LETTERS[order.index(0)]
        questions[key] = {"question": str(row["Question"]).strip(),
                          "choices": shuffled}
    return gt, questions


def aime():
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    gt, questions = {}, {}
    for i, row in enumerate(ds):
        key = f"aime_2024-{i + 1}"
        # AIME answers are written zero-padded (023); a model replies 23.
        # Both sides are normalised to the plain integer so exact match is
        # comparing the number, not its formatting.
        # Stored as an int, not a string: the contribution CSV is read with
        # pandas, which parses a numeric column to int64, and comparing that
        # against a string ground truth silently scores every row wrong.
        gt[key] = int(str(row["answer"]).strip())
        questions[key] = {"question": str(row["problem"]).strip(),
                          "choices": None}
    return gt, questions


all_q = {}
for name, prop, fn in [("gpqa_diamond", "test_quiz", gpqa),
                       ("aime_2024", "test_quiz", aime)]:
    gt, questions = fn()
    stem = f"{name}_{prop}"
    path = os.path.join(OUT_BENCH, stem + ".json.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(stem + ".json", json.dumps({"train": {}, "test": gt}))
    all_q[name] = questions
    print(f"  {name}: {len(gt)} items -> {path}")
    print(f"    answers: {sorted(set(gt.values()))[:6]}")

json.dump(all_q, open(OUT_Q, "w"))
print(f"  questions -> {OUT_Q} (GPQA option order seed={SEED})")
