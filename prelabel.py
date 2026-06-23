"""
prelabel.py — automated AI pre-labeling for the TakeMeter dataset.

Reads comments_to_label.csv, asks an LLM (Groq) to suggest ONE label per
comment using your rubric, and writes comments_prelabeled.csv for you to
REVIEW. Caches suggestions by comment_id, so re-runs only call the API for
comments that aren't labeled yet (including ones that previously FAILED).

This is pre-labeling WITH review (assignment-sanctioned). You MUST read and
correct every row — the model is a first pass, never the ground truth.

SETUP:
  pip install groq python-dotenv
  Put your key in the .env file in this folder:
      GROQ_API_KEY=your_actual_key
  (free key: https://console.groq.com/keys)

RUN:
  python prelabel.py

RATE LIMITS (important):
  Groq's free tier caps tokens-per-day PER MODEL. If one model hits its cap
  (HTTP 429), switch MODEL below to a different one — it has its own separate
  daily bucket, so you can keep going immediately instead of waiting 24h.
  llama-3.1-8b-instant has a larger free bucket and is fine for this task.

Failed/rate-limited comments are cached as "" and WILL be retried on the next
run. Your manual edits to `label` in comments_prelabeled.csv are PRESERVED
across re-runs (any label that differs from ai_prelabel is treated as review).
"""

import os
import csv
import json
import time

# Load GROQ_API_KEY from the .env file in this folder (falls back to the shell
# environment if python-dotenv isn't installed).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

INPUT = "comments_to_label.csv"
OUTPUT = "comments_prelabeled.csv"
CACHE = "prelabel_cache.json"
# Per-model daily token caps on the free tier. If you hit 429, switch model:
#   "llama-3.1-8b-instant"      <- larger free bucket, fast, good enough here
#   "llama-3.3-70b-versatile"   <- sharper but small daily cap (~100k tokens)
MODEL = "llama-3.1-8b-instant"
SLEEP = 0.3
LABELS = ["analysis", "hot_take", "reaction", "exclude"]

SYSTEM_PROMPT = """You label comments from r/worldcup (FIFA World Cup discussion) by discourse type. Assign EXACTLY ONE label.

analysis — makes a claim about the game/teams and supports it with a causal mechanism, specific tactical observation, historical comparison, or verifiable stat. The evidence does real argumentative work: it would still stand if you removed the opinion framing.
Example: "Ecuador averaged under 1 goal per match in qualifiers; they aren't a team that wins by wide margins."

hot_take — a confident opinion or verdict asserted WITHOUT real supporting evidence. Hedged or polite tone still counts; what matters is opinion-without-argument.
Example: "This is the most overrated France squad in years."

reaction — an on-topic response to a match event with little/no argument; a feeling or impression in the moment (emotion optional).
Example: "NOOO HOW DID HE MISS THAT"

exclude — NOT match discourse: off-topic chatter, jokes, bot messages, genuine questions, pure agreement with nothing under it, rule/format facts, or meta-argument about the thread itself.
Example: "anyone else's stream lagging?"

Decision rules:
- analysis vs hot_take: is the evidence doing real argumentative work, or is it decorative/cherry-picked? Real work -> analysis; decorative -> hot_take.
- analysis vs reaction: explains WHY (a mechanism) -> analysis; only describes WHAT happened -> reaction.
- hot_take vs reaction: a standing judgment you could state any day -> hot_take; tied to a just-happened moment -> reaction.
- A rhetorical question that embeds a claim -> label by the embedded claim. A genuine information-seeking question -> exclude.
- If a comment spans two types, label by its PRIMARY function. If it is a genuine 50/50 split with no real soccer substance -> exclude.

Respond with ONLY one word: analysis, hot_take, reaction, or exclude. No punctuation, no explanation."""


def classify(client, text):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Comment:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip().lower()
        for lab in sorted(LABELS, key=len, reverse=True):
            if raw == lab or lab in raw:
                return lab
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_reviewed():
    """Preserve human edits: any existing label != ai_prelabel is your review."""
    reviewed = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("comment_id", "")
                lab = (r.get("label") or "").strip()
                ai = (r.get("ai_prelabel") or "").strip()
                if cid and lab and lab != ai:
                    reviewed[cid] = lab
    return reviewed


def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        raise SystemExit("Set GROQ_API_KEY in the .env file (see setup at top of file).")
    client = Groq(api_key=key)

    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cache = load_json(CACHE)        # comment_id -> ai label ("" = needs retry)
    reviewed = load_reviewed()      # comment_id -> your corrected label

    new_calls = 0
    rate_limited = 0
    for i, r in enumerate(rows):
        cid = r.get("comment_id", "") or f"row{i}"
        # Retry anything not yet successfully labeled (missing OR empty "").
        if not cache.get(cid):
            label = classify(client, r.get("text", ""))
            cache[cid] = label or ""
            if not label:
                rate_limited += 1
            new_calls += 1
            if new_calls % 10 == 0:
                print(f"  attempted {new_calls} (still-unlabeled: {rate_limited})...")
                with open(CACHE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            time.sleep(SLEEP)

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    fieldnames = ["text", "label", "notes", "ai_prelabel", "ai_prelabeled",
                  "thread_type", "comment_id"]
    out = []
    for i, r in enumerate(rows):
        cid = r.get("comment_id", "") or f"row{i}"
        ai = cache.get(cid, "")
        label = reviewed.get(cid, ai)   # keep your edit if you made one
        out.append({
            "text": r.get("text", ""),
            "label": label,
            "notes": "" if ai else "NOT LABELED (rate-limited) — rerun or label by hand",
            "ai_prelabel": ai,
            "ai_prelabeled": "TRUE" if ai else "FALSE",
            "thread_type": r.get("thread_type", ""),
            "comment_id": r.get("comment_id", ""),
        })

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)

    counts = {}
    for r in out:
        counts[r["label"] or "UNLABELED"] = counts.get(r["label"] or "UNLABELED", 0) + 1
    labeled = sum(1 for r in out if r["label"])
    print(f"\nWrote {OUTPUT}: {len(out)} rows, {labeled} labeled this+prior runs")
    print("Label counts:", counts)
    if counts.get("UNLABELED"):
        print(f"\n{counts['UNLABELED']} still unlabeled (rate limit). Switch MODEL at top")
        print("of this file to llama-3.1-8b-instant (or wait for reset) and rerun.")
    print("\nNEXT:")
    print("  1. Open comments_prelabeled.csv and review EVERY row; correct `label`.")
    print("  2. Delete `exclude` and any UNLABELED rows before training.")
    print("  3. Report your override rate (label != ai_prelabel) in AI-usage disclosure.")


if __name__ == "__main__":
    main()
