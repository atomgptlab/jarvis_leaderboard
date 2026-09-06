"""GPQA-diamond (multiple choice) and AIME 2024 (integer answer), 0-shot.

AIME needs room to work: unlike a multiple-choice letter, the answer is the
end of a derivation, so the reply is not capped at a few tokens and the final
integer is taken from the end of whatever comes back.
"""
import asyncio, json, os, re, sys
import httpx

QUESTIONS, WHICH, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BASE = os.environ.get("ATOMSH_API_BASE", "https://atomgpt.org") + "/api"
TOKEN = json.load(open(os.path.expanduser(
    "~/.config/atomsh/auth.json")))["access_token"]
LETTERS = "ABCD"
data = json.load(open(QUESTIONS))[WHICH]

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
todo = [k for k in data if k not in done]
print(f"{WHICH}: total {len(data)} todo {len(todo)}", flush=True)


COT = os.environ.get("COT", "0") == "1"
# A reasoning model spends its budget before it writes an answer, so the cap
# has to be generous or the reply is truncated mid-thought and scores wrong.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))


def build(item):
    if item["choices"]:
        if COT:
            # GPQA is a reasoning benchmark; published scores let the model
            # work before answering. Forcing an immediate letter measures
            # recall, not the thing the benchmark is for.
            lines = [item["question"], ""]
            lines += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])]
            lines += ["", "Think it through, then end your reply with the "
                      "final choice on its own line as: ANSWER: <letter>"]
            return "\n".join(lines), MAX_TOKENS
        lines = ["Answer this multiple choice question.",
                 "Reply with only the letter of the correct answer.", "",
                 item["question"]]
        lines += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])]
        lines.append("Answer:")
        return "\n".join(lines), 12
    return (item["question"] + "\n\nThink it through, then end your reply with "
            "the final integer answer on its own line as: ANSWER: <integer>"), MAX_TOKENS


def parse(text, mc):
    if not text:
        return "X"
    if mc:
        m = re.search(r"ANSWER:\s*\(?([ABCD])\b", text, re.I)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([ABCD])\b", text.strip().upper())
        return m.group(1) if m else "X"
    # -1 marks "no answer" for integer benchmarks: it can never be correct,
    # and it keeps the CSV column numeric. A string marker would turn the
    # column into text and silently mismatch every integer key.
    m = re.search(r"ANSWER:\s*(-?\d+)", text, re.I)
    if not m:
        nums = re.findall(r"-?\d+", text)
        return str(int(nums[-1])) if nums else "-1"
    return str(int(m.group(1)))


async def one(client, sem, key, item, out, lock):
    async with sem:
        prompt, max_tok = build(item)
        mc = item["choices"] is not None
        body = {"model": MODEL, "max_tokens": max_tok, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        text = ""
        for attempt in range(5):
            try:
                r = await client.post(f"{BASE}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {TOKEN}"},
                                      timeout=600)
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt * 2)); continue
                if r.status_code >= 400:
                    break
                text = r.json()["choices"][0]["message"].get("content") or ""
                break
            except (httpx.HTTPError, KeyError, ValueError):
                await asyncio.sleep(min(30, 2 ** attempt))
        async with lock:
            out.write(json.dumps({"id": key, "prediction": parse(text, mc)}) + "\n")
            out.flush()


async def main():
    sem = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", "8")))
    lock = asyncio.Lock()
    with open(OUT, "a") as out:
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(one(client, sem, k, data[k], out, lock)
                                   for k in todo))
asyncio.run(main())
print("DONE", flush=True)
