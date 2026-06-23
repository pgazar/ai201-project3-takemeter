# TakeMeter — Planning

A fine-tuned classifier that measures discourse quality in r/worldcup by sorting comments into three kinds of contribution: reasoned analysis, confident-but-unsupported hot takes, and in-the-moment reactions. This document locks the design decisions before annotation begins, because the labels — not the training — are the hard part of this project.

---

## 1. Community

**Choice:** [r/worldcup](https://www.reddit.com/r/worldcup/), the main Reddit community for FIFA World Cup discussion (~1M members), sampled during and around the 2026 tournament.

**Why this community.** It produces a high volume of varied discourse in a short window, which is exactly what a classification task needs. Crucially, the *quality* of that discourse varies enormously: the same match thread contains rigorous tactical breakdowns, sweeping opinions with no support, and pure emotional venting — often within seconds of each other. That natural spread is what makes the three labels learnable. A community where everyone wrote the same way (all analysis, or all noise) would give the model nothing to distinguish. Soccer also has a deep tactical vocabulary (pressing, shape, transitions, xG) that gives genuine analysis a recognizable texture distinct from reactions, while still generating a flood of low-effort takes during big moments.

**Why it suits the task specifically.** The distinctions are not invented by me — they mirror a quality judgment the community already makes. Fans upvote tactical insight and scroll past or downvote low-effort hot takes and noise. Separating reasoning from assertion from raw reaction is therefore modeling a behavior that real users perform constantly, which is what makes a classifier here potentially *useful* rather than merely academic.

---

## 2. Labels

Three labels, plus an **exclude** bucket for content that is not match discourse at all (off-topic chatter, genuine questions, news links, stream complaints). The exclude bucket is dropped from the dataset, not labeled.

### `analysis`
A comment that makes a claim about the game and supports it with a causal mechanism, specific tactical observation, or verifiable statistic — something another person could engage with or dispute on substance.

- "Morocco sat in a 4-5-1 and forced Spain wide all game — Spain had 70% possession but it was all in front of the block, which is why they only managed 2 shots on target."
- "Brazil's fullbacks pushed so high it left the counter channel wide open, and that's exactly where Croatia got the equalizer."

### `hot_take`
A bold, confident opinion or verdict asserted *without* real supporting evidence — the claim might be true, but the comment asserts rather than argues. (Hedging or a polite, conversational tone does not disqualify a hot take: what matters is opinion-without-argument, not how loud it is.)

- "This is the most overrated France squad in years, full stop."
- "We won't win. We never win prizes since '88 but other teams still hate to play against us." *(cites a historical fact, but it's decorative — it props up the verdict without reasoning about this squad)*

### `reaction`
An immediate, on-topic response to a match event with little or no argument — expressing a feeling or impression in the moment.

- "NOOO HOW DID HE MISS THAT 😭😭"
- "Sweden never gave up, have to hand it to them — all Swedish, no Finnish." *(describes the flow of the game, but explains no mechanism, so it is description, not analysis)*

---

## 3. Hard edge cases and how I'll handle them

The labels are made mutually exclusive by a **waterfall decision flow**: check `analysis` first, then `hot_take`, then `reaction`; the first test a comment passes is its label. Six boundary rules resolve the genuine ambiguities.

**Boundary 1 — `analysis` vs `hot_take` (the evidence test).** Strip out the opinion framing and ask whether the evidence still stands as reasoning. If it does real argumentative work → `analysis`. If the stat or fact is cherry-picked or decorative — there to sound credible, not to reason → `hot_take`.

**Boundary 2 — `hot_take` vs `reaction` (standing vs moment).** A standing judgment you could state any day ("X is overrated") → `hot_take`. A response to something that just happened, about feeling ("that call was a disgrace!") → `reaction`.

**Boundary 3 — `reaction` vs exclude (on-topic vs chatter).** An on-topic response to the match being discussed, no argument → `reaction`, even if calm and flat ("expected the Dutch to struggle, turns out I was wrong"). Off-topic or personal chatter not about the match's substance → exclude ("I have tickets to that match!"). Reaction does *not* require strong emotion — only an in-the-moment response to this match with no argument.

**Boundary 4 — `analysis` vs `reaction` (WHY vs WHAT — the most common hard case).** Explains *why* something happened, with a mechanism or specific evidence → `analysis`. Merely describes *what* happened, however articulately → `reaction`. **Description is not analysis**; a polished, fair-minded summary of the game flow with no causal reasoning is still a reaction.

**Boundary 5 — questions (genuine vs rhetorical).** A *genuine information-seeking* question ("what channel is this on?", "why was that called offside?") → exclude. A *rhetorical* question that embeds a tactical argument or verdict → label by the embedded content (`analysis` or `hot_take`). Test: remove the question mark; if a standing claim is left, label the claim. *(Added after the boundary-post stress test in §8 surfaced this gap.)*

**Boundary 6 — multi-topic comments (one comment, two classes).** When a single comment contains material for two classes (e.g. a mild match description *and* a separate unsupported opinion about tournament format), label it by its **primary function** — the dominant content by weight and emphasis. Only fall back to `exclude` if it is a genuine 50/50 split with no dominant function, because a muddy label teaches the model noise. Also: agreement-wrapped comments split by what is *underneath* the agreement — "I concur, USA's around 12th" has a claim under it (`hot_take`), while "Correct, let's build confidence" has nothing under it (`exclude`). *(Added after multi-topic and agreement-wrapped comments recurred during annotation.)*

**Tiebreaker.** When a comment has elements of two classes (e.g. an emotional verdict like "Mbappé is WASHED 😤"), label it by what it is *primarily* doing, and when in doubt follow the waterfall order: `analysis` beats `hot_take` beats `reaction`.

**The most genuinely ambiguous type** I expect to hit repeatedly is the `analysis`/`hot_take` boundary: a comment that includes a real-sounding stat but uses it to prop up a verdict rather than to reason. My standing rule for these is Boundary 1's strip-the-opinion test, applied consistently. I will log every comment I hesitate on; if the *same kind* of hard case recurs, that is the signal to add another boundary rule before scaling to 200 — a one-off oddity is not.

**Surface-feature discipline.** CAPS, emoji, exclamation marks, length, and swearing *correlate* with the labels but are never the criteria. A short, sharp tactical comment is still `analysis`; a long, evidence-free rant is still `hot_take`. Labeling on surface features teaches the model "length," not discourse quality — the exact failure this project is designed to expose.

---

## 4. Data collection plan

**Source and method.** Comments collected from r/worldcup. Because Reddit blocks unauthenticated scripts, raw text is gathered by saving each thread's public `.json` from a logged-in browser and parsing the saved files locally (`parse_json.py`). Comments are then AI pre-labeled (`prelabel.py`, Groq) and I review and correct every row by hand — see §8 for the disclosed workflow. To get class variety rather than a flood of one type, I sample from three different *kinds* of threads:

- **Post-match threads** — rich in `reaction`, scattered `hot_take`.
- **Tactical / discussion posts** ("How did X shut down Y", formation debates) — the primary source of `analysis`, which is the scarcest class.
- **Debate / opinion posts** ("Most overrated team left?", ranking arguments) — a `hot_take` hotspot.

The parser filters out deleted/removed comments, ultra-short noise (<15 chars), and copypasta (>1200 chars), dedupes by comment ID, and caps comments per thread so no single thread floods the set. Output is `comments_to_label.csv`; the reviewed labels live in `comments_prelabeled.csv` with `text` / `label` / `notes` columns (matching the training notebook), one row per comment.

**Target volume.** ~200+ labeled in-class comments, aiming for rough balance across the three classes. Because `reaction` is over-represented in live match threads and `analysis` is scarce, I sample tactical/discussion threads heavily for `analysis` and post-match threads for `reaction`. Since `exclude` rows are deleted before training, I over-collect (~350 raw) to net well over 200 in-class examples.

**If a label is underrepresented.** First, do targeted re-collection from the thread type richest in that class (more tactical posts for `analysis`, more post-match threads for `reaction`) rather than padding from whatever is easiest. Second, if a class stays scarce, lower the overall target to preserve balance rather than train on a lopsided set. Per the assignment's check, if any single label exceeds 70% of the in-class dataset I collect more of the underrepresented labels before training. I do **not** fill gaps with AI-generated posts in the training data — synthetic posts carry model priors and would corrupt the real-distribution signal. Any residual imbalance is reported honestly and handled in the metrics (§5), not hidden.

**Annotation hygiene.** Lock the rubric before full annotation; read every comment rather than skimming (even the AI-pre-labeled ones); keep a running pause-log of hard cases in the `notes` column; re-label a random 10% at the end to measure self-consistency. The training notebook holds out a random, stratified 15% test set that is not used for training.

---

## 5. Evaluation metrics

**Why accuracy alone is misleading.** This is an imbalanced three-class problem. If `reaction` is the plurality class, a model that simply predicts "reaction" every time can post a deceptively high accuracy while being useless at the distinctions that matter. Accuracy rewards the majority class and hides failure on the rare, valuable ones. (Note: the starter notebook prints accuracy as its headline number, but its `classification_report` also reports per-class and macro-averaged F1 — I read my primary metric off that report, below.)

**Primary metric — macro-F1.** The unweighted mean of per-class F1. It treats all three classes equally regardless of frequency, so the model cannot coast on the majority class. I deliberately choose *macro*-F1 over *weighted*-F1 here: weighted-F1 would re-import the very frequency bias I'm trying to control for, since it lets the dominant `reaction` class dominate the headline number.

**Per-class precision / recall / F1.** Reported for all three classes, because different errors carry different costs for *this* task:
- `analysis` is the highest-value class for a discourse-quality tool. **Precision on `analysis`** is the most deployment-critical number — if hot takes and reactions get mislabeled as analysis, the tool promotes noise and loses user trust.
- **Recall on `analysis`** matters too: silently burying genuine insight defeats the tool's purpose.
- So **`analysis` F1** is the key single-class metric, and the `analysis`↔`hot_take` boundary is where I expect — and will scrutinize — the most error.

**Confusion matrix — the diagnostic centerpiece.** This is how I separate *what the model captured* from *what I intended*. The specific cells tell a story: heavy `analysis`↔`reaction` confusion means the model failed the WHY-vs-WHAT distinction (likely learned length/surface features instead); heavy `analysis`↔`hot_take` confusion means it failed the evidence test. Reading the matrix is how I diagnose the actual learned behavior.

**Fine-tuned vs zero-shot baseline.** The required comparison. I report macro-F1 and per-class F1 for both the fine-tuned model and a zero-shot prompted baseline, and I report the result honestly even if fine-tuning's margin over prompting is small. If the baseline is nearly as good, that is a finding, not a failure. (Caveat noted in §8: because AI pre-labeling reaches the test set, the zero-shot baseline may be modestly flattered.)

**Qualitative error analysis.** Read the misclassified examples and check explicitly for the length/surface-feature shortcut (does it mislabel short analysis as reaction, or long rants as analysis?). This is the "what did it actually learn" check that no aggregate number can give.

**Annotation-consistency ceiling.** Since I am the sole annotator, my 10% re-label agreement sets a realistic upper bound: the model cannot be expected to be more consistent than the labels it learned from. I'll report this number as context for the model's scores.

---

## 6. Definition of success

Stated as objective thresholds on the held-out test set so the outcome is unambiguous.

**Genuinely useful (project bar):**
- **Macro-F1 ≥ 0.70.** Credible for a subjective three-class task on ~200 examples; well above the ~0.33 a random/majority predictor would reach on macro-F1.
- **Beats the zero-shot baseline on macro-F1 by ≥ 5 points.** Otherwise fine-tuning did not earn its keep and the honest conclusion is "prompting was enough."
- **No dead class:** `analysis` F1 ≥ 0.65 and no single class below 0.55 F1.

**Good enough for deployment (stricter bar):**
- **Precision on `analysis` ≥ 0.75.** For a real community tool that surfaces high-quality discourse, when it flags a comment as analysis it must usually be right, or users stop trusting it. I would accept lower *recall* on analysis as a tradeoff — missing some good posts is less damaging than promoting noise.
- **Model macro-F1 within striking distance of my self-consistency ceiling**, confirming the limiting factor is annotation quality, not the model.

---

## 7. Evaluation plan review

Are these criteria specific enough to objectively judge at the end? Yes. Each is a named metric with a numeric threshold on a held-out test set, so pass/fail is determinable without further interpretation: macro-F1 against 0.70, a ≥5-point margin over baseline, per-class F1 floors, and precision-on-analysis against 0.75. The confusion matrix and error analysis are diagnostic, not pass/fail, and are labeled as such.

The one soft spot is honest to flag: the deployment bar leans on the self-consistency ceiling, which depends on the quality of my own annotation. If my 10% re-label agreement comes back low (say <0.80), that caps how meaningful any model score is, and the right response would be to tighten the rubric and re-annotate rather than celebrate a number. That contingency is built into the plan rather than discovered at the end.

---

## 8. AI Tool Plan

This project generates no application code. AI tools appear in three places in the workflow; I use all three, each logged so the AI's contribution stays auditable.

### (1) Label stress-testing — *before* annotating  *(used)*
Generate 5–10 posts that sit on a boundary between two labels and try to classify each cleanly under the rubric. A post that can't be classified cleanly means the definitions need tightening — fixed now, not after the full annotation. The eight generated posts and the definition fix they produced are below.

### (2) Annotation assistance — AI pre-labeling with full human review  *(used)*
- **Tool & workflow:** a Groq-hosted LLM (`llama-3.3-70b-versatile`, with `llama-3.1-8b-instant` as the fallback model once the 70B daily token cap was reached) is prompted with the rubric from §2–§3 and assigns one of `analysis | hot_take | reaction | exclude` to each comment (`prelabel.py`). The suggestion is written to the `ai_prelabel` column; I then read every row and record my decision in `label`. **Only my reviewed `label` is used for training** — the model is a first pass, never the ground truth.
- **Disclosure & accountability:** every pre-labeled row is flagged `ai_prelabeled = TRUE`. I report (a) the share of the dataset that was pre-labeled and (b) my **override rate** — how often my final `label` differs from `ai_prelabel` — as concrete evidence I reviewed rather than rubber-stamped. The model is weakest exactly on my scarce, high-value class (terse, evidence-based rebuttals that read like quips but are really `analysis`), so those rows get extra scrutiny on review.
- **Known limitation (disclosed, not hidden):** the training notebook makes its own random, stratified 70/15/15 split, so I cannot quarantine the test set from pre-labeling — AI-suggested labels reach the test set too. This can modestly *flatter the zero-shot baseline*, since that baseline shares the pre-labeler's priors. I mitigate it with genuine row-by-row review (high override rate), and I flag the caveat when reporting the fine-tuned-vs-zero-shot gap rather than presenting a small margin as clean.

### (3) Failure analysis — pattern-finding on wrong predictions  *(used)*
- **Workflow:** after evaluation, export the misclassified test examples (comment text, true label, predicted label) and give that list to an AI tool, asking it to identify patterns in the errors.
- **What I'll look for:** the length/surface-feature shortcut (are false-`analysis` errors mostly long comments? are short analyses misread as reactions?); which boundary fails most (is `analysis`↔`hot_take` the dominant confusion, meaning the evidence test wasn't learned?); and whether particular topics or phrasings cluster in the errors.
- **How I'll verify:** every pattern the AI proposes is a *hypothesis, not a finding*. I confirm each against the confusion matrix and by re-reading the actual misclassified comments myself before anything goes in the evaluation write-up. The AI narrows where to look; I confirm what is true.

### Boundary-post stress test (8 generated posts)

| # | Generated post | Boundary | Clean label? | Resolution |
|---|---|---|---|---|
| 1 | "Spain controlled the half because Rodri kept dropping between the CBs, giving them a 3v2 against Japan's press." | analysis / hot_take | ✅ `analysis` | Mechanism survives stripping opinion (B1). |
| 2 | "England's xG has been under 1.0 in three straight knockouts — they're just not a serious attacking team." | analysis / hot_take | ✅ `hot_take` (close) | Real stat, but the verdict outruns it; deployed as decorative support for a sweeping claim (B1). |
| 3 | "OFF THE POST AGAIN ARE YOU KIDDING ME 😭" | hot_take / reaction | ✅ `reaction` | In-the-moment emotion, no standing verdict (B2). |
| 4 | "Southgate out. Worst manager we've ever had." (posted after a sub) | hot_take / reaction | ✅ `hot_take` | Core is a standing verdict, only triggered by the moment (B2). |
| 5 | "Pretty even game tbh, both teams kind of cancelled each other out." | analysis / reaction | ✅ `reaction` | Describes WHAT, no mechanism (B4). |
| 6 | "anyone else's stream lagging or is it just me" | reaction / exclude | ✅ exclude | Off-topic, not match discourse (B3). |
| 7 | "Why do they keep lumping it long when both CBs are clearly more comfortable building from the back?" | exclude / analysis | ⚠️ **NOT clean** | Phrased as a question (→ old exclude rule), but embeds a real tactical argument. Prompted Boundary 5. |
| 8 | "Oh brilliant, let him stroll through the whole midfield. World-class defending. 🙄" | reaction / hot_take | ✅ `reaction` (watch) | Sarcasm carrying implied critique; resolves to `reaction` (in-the-moment, no actual argument), but sarcasm is a recurring watch-item. |

### Definition tightening prompted by the test

Post #7 did **not** classify cleanly, and that is the test working as intended. The original exclude rule ("questions → exclude") was too blunt: a *rhetorical* question can embed genuine analysis. This produced **Boundary 5** (§3): a genuine information-seeking question is excluded, but a rhetorical question that embeds an argument is labeled by its embedded content. Under that rule, Post #7 → `analysis`. Post #8 (sarcasm) stays `reaction` but is logged as a watch-item.

Two further boundaries emerged during live annotation (not from generated posts): **Boundary 5** finalized from the rhetorical-question case above, and **Boundary 6** from real multi-topic and agreement-wrapped comments (e.g. a comment mixing match reaction with tournament-format opinions, or "I concur, USA's ~12th"). Both are documented in §3.
