"""
parse_json.py — build the dataset from saved Reddit JSON files.

NO network, NO API, NO credentials. You save each thread's .json from your
logged-in browser (which isn't blocked), drop the files in ./raw_json/, and
this reads them locally into comments_to_label.csv for hand-labeling.

HOW TO GET THE JSON FILES:
  For each thread, open its URL with .json added before any "?", e.g.
    https://www.reddit.com/r/worldcup/comments/1ub4jdy/.../.json
  Then right-click -> Save As, into the raw_json/ folder.

  Name files so the thread type is recognizable (underscores optional):
    postmatch_1.json / post_match_1.json -> post_match
    tactical_1.json                      -> tactical
    debate_2.json                        -> debate
  Anything else is tagged "other".

RUN:
  python parse_json.py
  -> writes comments_to_label.csv, then fill in the `label` column by hand.

TUNING:
  PER_THREAD_LIMIT caps comments taken per thread (so one thread can't flood
  the set). Raise it to pull more per thread, lower it to pull fewer.
"""

import os
import csv
import glob
import json

RAW_DIR = "raw_json"
MIN_CHARS = 15
MAX_CHARS = 1200
PER_THREAD_LIMIT = 60      # max comments per thread (12 threads x 60 = up to 720 raw)


def clean(text: str) -> str:
    return " ".join(text.split())


def thread_type_from_name(filename: str) -> str:
    # normalize: lowercase, drop underscores/spaces/hyphens so "postmatch",
    # "post_match", "post-match" all match.
    base = os.path.basename(filename).lower()
    norm = base.replace("_", "").replace("-", "").replace(" ", "")
    if norm.startswith("postmatch"):
        return "post_match"
    if norm.startswith("tactical"):
        return "tactical"
    if norm.startswith("debate"):
        return "debate"
    return "other"


def walk(children, thread_type, seen, rows, taken):
    for child in children:
        if taken[0] >= PER_THREAD_LIMIT:
            return
        if child.get("kind") != "t1":      # t1 = comment
            continue
        data = child.get("data", {})
        body = (data.get("body") or "").strip()
        cid = data.get("id")
        if body and body not in ("[deleted]", "[removed]") and cid not in seen:
            if MIN_CHARS <= len(body) <= MAX_CHARS:
                seen.add(cid)
                rows.append({
                    "text": clean(body),
                    "label": "",
                    "notes": "",
                    "thread_type": thread_type,
                    "comment_id": cid,
                })
                taken[0] += 1
        replies = data.get("replies")
        if isinstance(replies, dict):
            walk(replies.get("data", {}).get("children", []),
                 thread_type, seen, rows, taken)


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        print(f"No .json files found in {RAW_DIR}/ — save your thread JSON there first.")
        return

    seen = set()
    rows = []
    for path in files:
        thread_type = thread_type_from_name(path)
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ! {name} — not valid JSON ({e}).")
            print(f"      The browser likely saved it as HTML. Re-open the .json URL")
            print(f"      and use Save As -> Format: Page Source (plain text).")
            continue
        except Exception as e:
            print(f"  ! {name} — couldn't read ({e})")
            continue
        try:
            comments = payload[1]["data"]["children"]
        except (IndexError, KeyError, TypeError):
            print(f"  ! {name} — JSON parsed but not a thread (wrong page saved?)")
            continue
        taken = [0]
        walk(comments, thread_type, seen, rows, taken)
        print(f"  {name} ({thread_type}): +{taken[0]} comments")

    if not rows:
        print("\nNo comments extracted — see messages above.")
        return

    with open("comments_to_label.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_type = {}
    for r in rows:
        by_type[r["thread_type"]] = by_type.get(r["thread_type"], 0) + 1
    print(f"\nWrote {len(rows)} comments to comments_to_label.csv")
    print("By thread type:", by_type)
    print("Now open the CSV and fill in the `label` column by hand.")


if __name__ == "__main__":
    main()
