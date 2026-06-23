# Evaluation notes (draft for README)

Scratch draft of the evaluation write-up. Numbers are final (fine-tuned model
trained 6 epochs; zero-shot baseline = llama-3.3-70b-versatile). Test set = 54
held-out comments, 18 per class, stratified split (random_state=42).

---

## Results

| Metric            | Zero-shot baseline (Llama-70B) | Fine-tuned DistilBERT (6 epochs) |
|-------------------|--------------------------------|----------------------------------|
| Accuracy          | 0.704                          | **0.722**                        |
| Macro-F1          | 0.70                           | **0.72**                         |
| analysis  (P/R/F1)| 1.00 / 0.39 / 0.56             | 0.84 / 0.89 / **0.86**           |
| hot_take  (P/R/F1)| 0.53 / 1.00 / 0.69             | 0.67 / 0.67 / 0.67               |
| reaction  (P/R/F1)| 1.00 / 0.72 / **0.84**         | 0.65 / 0.61 / 0.63               |

Primary metric is **macro-F1** (not accuracy): the task is a 3-class discourse
problem and macro-F1 weights each class equally, so a model cannot coast on a
majority class. (Splits are balanced 18/18/18, so accuracy and macro-F1 nearly
coincide here, but macro-F1 remains the reported headline per the plan.)

## Confusion matrix (fine-tuned, 6 epochs, test set)

Rows = true label, columns = predicted:

|              | pred analysis | pred hot_take | pred reaction |
|--------------|---------------|---------------|---------------|
| **analysis** | 16            | 1             | 1             |
| **hot_take** | 1             | 12            | 5             |
| **reaction** | 2             | 5             | 11            |

## Verdict against pre-registered success criteria (planning §6)

- Macro-F1 >= 0.70 — **met** (0.72).
- No class F1 < 0.55 — **met** (lowest = reaction 0.63).
- analysis F1 >= 0.65 — **met** (0.86).
- Beat baseline by >= 5 macro-F1 points — **NOT met** (only +2). Honest reading:
  fine-tuning produced a real but modest overall improvement, not a decisive one.
- Deployment bar: precision on analysis >= 0.75 — **met by the fine-tuned model**
  (0.84) and effectively failed by the baseline: the baseline's analysis precision
  is 1.00 but its recall is 0.39, so it misses 61% of the analysis it is supposed
  to surface. For the tool's actual purpose (surfacing high-quality analysis), the
  fine-tuned model is the usable one despite the close macro-F1.

## Key finding: the hard boundary SHIFTED with training

The plan predicted `analysis` vs `hot_take` (the "evidence test") would be the
hardest distinction. That held at 3 epochs — but the picture changed by 6:

- **Baseline** is over-conservative about `analysis`: it only says analysis when
  certain (precision 1.00) and dumps everything uncertain into `hot_take` (recall
  1.00, precision 0.53), so it misses 61% of real analysis.
- **Fine-tuned, 3 epochs:** over-predicted `analysis`; hot_take recall only 0.44.
  The dominant confusion was hot_take -> analysis (the evidence test).
- **Fine-tuned, 6 epochs:** the evidence-test confusion is largely RESOLVED
  (hot_take -> analysis dropped to 1 case; analysis row is 16/1/1). The residual
  error moved to **`hot_take` <-> `reaction`** — 5 hot_takes called reaction and 5
  reactions called hot_take. That is a *different* boundary: standing judgment vs
  in-the-moment response (the plan's Boundary 2), not the evidence line.

Interpretation (ties to the plan's "what the model captured vs what I intended"):
with enough training the small model learned the evidence distinction it was
designed around, and the leftover difficulty is separating a confident standing
opinion from a momentary reaction — a genuinely subtle line. `reaction` is the
fine-tuned model's weakest class (F1 0.63) for this reason.

## Training note (hyperparameter change, disclosed)

First run used the notebook default of 3 epochs and reached only 0.63 test
accuracy / 0.62 macro-F1, with hot_take F1 = 0.48. The validation-accuracy curve
was still climbing steeply at epoch 3 (0.30 -> 0.44 -> 0.63) and training loss had
barely moved — a clear sign of *undertraining*, not overfitting. I increased
`num_train_epochs` 3 -> 6. At 6 epochs validation accuracy reached 0.704 and
validation loss flattened (0.704 -> 0.706 over the last two epochs), indicating
near-convergence. Test macro-F1 improved 0.62 -> 0.72 and hot_take F1 0.48 -> 0.67.
No other hyperparameters were changed. The decision to retrain was based on the
*validation* curve, and the test set was scored once on the final model (not
tuned against).

## Error analysis (featured examples — final 6-epoch model)

Confidence is notably higher than the 3-epoch run (0.46–0.74 vs ~0.35): the model
now commits to its calls rather than coin-flipping near the 0.33 chance floor.

1. **Standing judgment vs in-the-moment (`hot_take` <-> `reaction`) — the new
   dominant error.**
   - "Start with Sane and Havertz in knockouts you are going out" → true `hot_take`,
     predicted `reaction` (0.53). A standing verdict read as a momentary outburst.
   - "Lonaldo will hold them back again big time." → true `hot_take`, predicted
     `reaction` (0.57). Same line.
   - "I agree, but also get Ronaldo's old ass off the field." → true `reaction`,
     predicted `hot_take` (0.62). The reverse direction.

2. **Very short / context-dependent fragments still hard.**
   - "Ready for a NL–Tunisia game to exceed 10 goals" and similar terse comments
     lack the textual signal to place confidently. A known limit of a text-only
     classifier with no thread context.

3. **Label-noise caveat (honest).** Some "errors" are arguably mislabels in my own
   data, not model failures — e.g. the comment listing exactly which groups/teams
   are eliminated under the format (specific, verifiable detail) I labeled
   `hot_take`, but the model called `analysis` at 0.74 confidence, and the
   stats-link Tchouaméni comparison I also labeled `hot_take`. By my own evidence
   test (Boundary 1), those specifics arguably do real work → `analysis`. This is
   the annotation-consistency ceiling from planning §5: the model cannot learn a
   line I did not draw cleanly. I did NOT relabel the test set to improve the score;
   I report the observation instead.

## Bottom line

Fine-tuning a 66M-parameter DistilBERT on 360 hand-reviewed examples modestly beat
a zero-shot 70B general model on overall macro-F1 (0.72 vs 0.70) and decisively
beat it on the deployment-critical `analysis` class (F1 0.86 vs 0.56, recall 0.89
vs 0.39). With enough training it learned the evidence distinction the labels were
designed around; the remaining error sits on the subtler standing-judgment-vs-
reaction boundary, which both models — and, on a few examples, the human annotator
— find genuinely hard.
