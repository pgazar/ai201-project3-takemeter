"""
sample_for_review.py — take a balanced subset to hand-review.

You over-collected (~483 in-class). Reviewing all of them is unnecessary.
This drops `exclude` rows and randomly samples up to N_PER_CLASS comments per
class, so you review a balanced, manageable set that still clears the 200 min.

RUN:
  python sample_for_review.py
  -> writes comments_review.csv  (review the `label` column here, then this is
     the file you upload to the notebook)

Random sampling uses a fixed seed so re-runs give the same set.
"""

import csv
import random

INPUT = "comments_prelabeled.csv"
OUTPUT = "comments_review.csv"
N_PER_CLASS = 120          # comments per class (caps a class if it has fewer)
CLASSES = ["analysis", "hot_take", "reaction"]
SEED = 42


def main():
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class = {c: [] for c in CLASSES}
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab in by_class:
            by_class[lab].append(r)

    random.seed(SEED)
    picked = []
    for c in CLASSES:
        pool = by_class[c]
        random.shuffle(pool)
        picked.extend(pool[:N_PER_CLASS])

    random.shuffle(picked)  # mix classes so review isn't monotonous

    fieldnames = ["text", "label", "notes", "ai_prelabel", "ai_prelabeled",
                  "thread_type", "comment_id"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    counts = {}
    for r in picked:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"Wrote {OUTPUT}: {len(picked)} comments")
    print("Per-class:", counts)
    print("\nNEXT: review the `label` column in comments_review.csv (read every row),")
    print("then it's ready for the notebook. ai_prelabel is your AI-vs-final record.")


if __name__ == "__main__":
    main()
