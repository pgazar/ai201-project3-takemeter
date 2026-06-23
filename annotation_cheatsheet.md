# Annotation Cheat-Sheet — r/worldcup Discourse Quality

**Unit of analysis:** individual comments
**Classes:** `analysis` · `hot_take` · `reaction`
**Excluded (do NOT keep):** off-topic chatter, genuine questions, news links, stream complaints → label `exclude`, then DELETE before training

---

## The 3 labels (one line each)

- **`analysis`** — makes a claim and supports it with a causal mechanism, specific tactical observation, or verifiable stat. Something you could dispute on substance.
- **`hot_take`** — a confident opinion/verdict asserted *without* real evidence. (Hedged or polite tone still counts — what matters is opinion-without-argument, not volume.)
- **`reaction`** — an on-topic response to a match event with no argument; a feeling/impression in the moment.

---

## Quick decision flow (waterfall — first match wins)

1. Specific, verifiable evidence + reasoning (a mechanism, a "because")? → **`analysis`**
2. Else, a confident claim/verdict with no real evidence? → **`hot_take`**
3. Else, an on-topic response to a match event with no argument? → **`reaction`** (emotion optional)
4. Else, off-topic / no substantive content about the soccer? → **`exclude`**

---

## The six boundary rules

**Boundary 1 — `analysis` vs `hot_take`** (evidence test): strip the opinion framing; does the evidence still stand as reasoning? Yes → `analysis`. Cherry-picked/decorative → `hot_take`.

**Boundary 2 — `hot_take` vs `reaction`** (standing vs moment): a standing judgment you could state any day → `hot_take`. A response to something that just happened, about feeling → `reaction`.

**Boundary 3 — `reaction` vs exclude** (on-topic vs chatter): on-topic response to the match, no argument → `reaction` (even if calm/flat). Off-topic / personal chatter → `exclude`. Reaction does NOT require strong emotion.

**Boundary 4 — `analysis` vs `reaction`** (WHY vs WHAT): explains *why* (mechanism/evidence) → `analysis`. Just describes *what* happened, however articulately → `reaction`. **Description is not analysis.**

**Boundary 5 — questions** (genuine vs rhetorical): genuine info-seeking question → `exclude`. Rhetorical question embedding an argument → label by the embedded content. Test: remove the question mark; if a claim is left, label the claim.

**Boundary 6 — multi-topic / agreement-wrapped comments**: one comment spanning two classes → label by its **primary function** (dominant content by weight). Genuine 50/50 split with no dominant function → `exclude`. Agreement-wrapped comments split by what's *underneath*: "I concur, USA's ~12th" → `hot_take` (claim under it); "Correct, let's build confidence" → `exclude` (nothing under it).

**Tiebreaker:** elements of two classes → label by what it's primarily doing; when in doubt follow the waterfall order (`analysis` > `hot_take` > `reaction`).

---

## ⚠️ Surface-feature warning

CAPS, emoji, exclamation marks, length, swearing **correlate** with labels but are **NOT** the criteria. A short sharp tactical comment is still `analysis`; a long evidence-free rant is still `hot_take`. Labeling on surface features teaches the model "length," not discourse quality.

---

## Example bank

### `analysis`
- "Morocco sat in a 4-5-1 and forced Spain wide — 70% possession but in front of the block, so only 2 shots on target."
- "That's offside — his shoulder was clearly past the last defender." *(in the moment, but a specific verifiable observation)*

### `hot_take`
- "Most overrated France squad in years, full stop."
- "We won't win — we never win prizes since '88." *(decorative historical fact propping up a verdict)*
- "I concur, USA's around 12th, hard to judge this early." *(hedged/polite, but still an opinion with no argument)*

### `reaction`
- "NOOO HOW DID HE MISS THAT 😭😭"
- "Sweden never gave up — all Swedish, no Finnish." *(describes the game, no mechanism)*

### `exclude`
- "I have tickets to that match!" *(off-topic personal chatter)*
- "Correct, let's build confidence." *(agreement with nothing underneath)*
- "anyone else's stream lagging?" *(off-topic)*

### Borderline — apply the rules
- "This ref is the worst in the tournament." → standing verdict → `hot_take` (B2)
- "Mbappé's been quiet because they play him left against deep blocks." → mechanism → `analysis` (B4)
- "Good first half, on our heels the second… also they should ditch the four-period format." → spans reaction + format opinions; format opinions dominate → `hot_take` (B6)

---

## Annotation hygiene

- **Lock this rubric** — no criteria changes mid-stream (these six are locked).
- **Read every comment** — no skimming.
- **Log hard cases** in the `notes` column (what it was, which labels, what you decided).
- **Re-label a random 10%** at the end to measure self-consistency.
- **Watch balance**: if any class tops 70%, collect more of the thin ones (analysis/hot_take).
- **Before upload**: delete all `exclude` rows; keep only `analysis`/`hot_take`/`reaction`.
