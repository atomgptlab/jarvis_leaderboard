"""0-shot MMLU against the AtomGPT endpoint.

Answers are checkpointed to JSONL as they arrive, so an interrupted run
resumes instead of paying for the same questions twice. A question that
cannot be parsed is recorded as a wrong answer rather than dropped: dropping
it would quietly shrink the denominator and flatter the score.
"""
import asyncio, json, os, re, sys, time
import httpx

QUESTIONS, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
BASE = os.environ.get("ATOMSH_API_BASE", "https://atomgpt.org") + "/api"
TOKEN = json.load(open(os.path.expanduser(
    "~/.config/atomsh/auth.json")))["access_token"]
LETTERS = "ABCD"

data = json.load(open(QUESTIONS))
done = {}
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done[r["id"]] = r["prediction"]
        except ValueError:
            pass
todo = [k for k in data if k not in done]
print(f"total {len(data)} | done {len(done)} | todo {len(todo)}", flush=True)


def prompt_for(item, key):
    subject = key.rsplit("-", 1)[0].split("-")[-1].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.",
             "Reply with only the letter of the correct answer.", "",
             item["question"]]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def parse(text):
    if not text:
        return "X"
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else "X"


async def one(client, sem, key, item, out, lock):
    async with sem:
        body = {"model": MODEL, "max_tokens": 12, "temperature": 0,
                "messages": [{"role": "user", "content": prompt_for(item, key)}]}
        text = ""
        for attempt in range(5):
            try:
                r = await client.post(f"{BASE}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {TOKEN}"},
                                      timeout=120)
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt * 2))
                    continue
                if r.status_code >= 400:
                    break
                text = (r.json()["choices"][0]["message"].get("content") or "")
                break
            except (httpx.HTTPError, KeyError, ValueError):
                await asyncio.sleep(min(30, 2 ** attempt))
        async with lock:
            out.write(json.dumps({"id": key, "prediction": parse(text)}) + "\n")
            out.flush()


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    t0 = time.time()
    with open(OUT, "a") as out:
        async with httpx.AsyncClient() as client:
            batch = 200
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                await asyncio.gather(*(one(client, sem, k, data[k], out, lock)
                                       for k in chunk))
                pct = (i + len(chunk)) / max(1, len(todo)) * 100
                print(f"  {i+len(chunk)}/{len(todo)} ({pct:.0f}%) "
                      f"{time.time()-t0:.0f}s", flush=True)

asyncio.run(main())
print("DONE", flush=True)
