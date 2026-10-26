# Induction heads in Pythia-160m: my own measurements (2026-10-23)
Code: induction.py (numpy forward pass, validated against HF transformers in run14), kcomp.py, sweep.sh. Raw outputs: run1–run4 txt.
Test: 100 random tokens (ids 1000–50000) repeated twice, T=200, no BOS. Induction score = mean attention from position i in copy 2
to i−S+1; prev-token score = mean attention to i−1. Copying = top-1 accuracy / median NLL on copy-2 targets.
Medians, because Pythia-160m's logits are very spread (std ≈80, offset ≈+806, cf. vocab-init), so a few confident misses dominate means.
**Validated 10-24 against HF transformers 5.20.2 (validate_hf.py, run14):** same weights and tokens; argmax agreement 100%, max |logit diff| 0.09 on a logit std of ~125, max |attention diff| 0.008, layer-4 residual max diff 0.0001. So the numpy pass is correct, and the huge logit spread (std ≈80–150) is real Pythia behaviour, not my bug.
Note: my test differs from Olsson's (25 tokens × 4 repeats + BOS; prev-token score on real text). Absolute values aren't comparable; the patterns are.
1 step = 1024 × 2048 = 2.1M tokens (Pythia batch).

## 1. Final model (step 143000, 16 sequences)
- Prev-token heads: L3H2 0.97, L7H7 0.83, L3H0 0.80.
- Induction heads: L4H6 0.97, L8H2 0.88, L4H10 0.85, L5H6 0.81, L5H0 0.79, L9H8 0.67 (then a continuum, as Olsson says; there's no clear threshold).
- Copy 2: top-1 0.879, median NLL 0.54 (copy 1: 13.0 ≈ chance or worse).
- Zero-ablation (copy-2 top-1; control = K random heads from outside the top-24 induction list):
| K | top-K induction | K random |
|---|---|---|
| 1 | 0.691 | 0.878 |
| 3 | 0.218 | 0.874 |
| 6 | 0.004 | 0.867 |
| 12 | 0.000 | 0.028 |
| 24 | 0.000 | 0.002 |
  Caveat: at K≥12 the random control also collapses. The control pool includes the prev-token heads (L3H2 etc.), and zero-ablating 12 heads is a big perturbation. K≤6 is the clean comparison: 6 induction heads remove copying completely, while 6 random heads cost 1 point.

## 2. Training sweep (8 sequences per checkpoint)
| step | tokens | best prev-token (L3H2) | best induction | copy-2 top-1 |
|---|---|---|---|---|
| 0–128 | ≤0.27B | 0.01–0.03 (chance ≈1/T) | 0.008 | 0.000 |
| 256 | 0.54B | 0.13 | 0.007 | 0.000 |
| 512 | 1.07B | 0.37 | 0.023 (L6H6) | 0.000 |
| 1000 | 2.1B | 0.94 | 0.97 (L4H10), L4H6 0.93 | 0.508 |
| 2000 | 4.2B | 0.96 | 0.98 (L4H6) | 0.803 |
| 4000 | 8.4B | 0.97 | 0.98 | 0.909 |
| 8000 | 16.8B | 0.98 | 0.98 | 0.948 |
| 143000 | 300B | 0.97 | 0.97 | 0.879 |
Findings:
- **Order: the prev-token head comes first.** L3H2 is already at 0.13 (step 256) and 0.37 (step 512), while every induction score is still ≈0. Induction appears only between 512 and 1000. That's what K-composition predicts (an induction head can't form before the signal it reads exists). The checkpoints are too coarse (512 → 1000) to see the jump itself.
- **Timing:** 1.1–2.1B tokens. Olsson et al. report 2.5–5B for their models; Yin & Steinhardt report ≈ step 1000 for Pythia. Consistent with the latter. Olsson's window differs (other models and data).
- **Two waves.** The first wave is layers 4–5 (L4H6, L4H10, L4H11, L5H6, L5H0), formed by step 1000. The layer 8–9 heads (L8H2, L9H8), strong in the final model (0.88, 0.67), are still weak at step 8000 (0.11). A second set forms later, between 17B and 300B tokens (see the late sweep).
- Copying peaks at step 8000 (0.948) and is LOWER at the end (0.879). That's a mild decline, consistent with Yin & Steinhardt's "plateau or slight decline". n=8 vs 16 sequences, so it's suggestive.

## 3. Causal test of K-composition (final model, 8 sequences): ablate prev-token heads, measure downstream induction attention
| ablated | L4H6 | L4H10 | L5H6 | L5H0 | L8H2 | L9H8 | copy top-1 |
|---|---|---|---|---|---|---|---|
| none | 0.967 | 0.832 | 0.824 | 0.774 | 0.869 | 0.669 | 0.895 |
| L3H2 only | 0.055 | 0.038 | 0.508 | 0.249 | 0.518 | 0.089 | 0.052 |
| L3H0 only | 0.723 | 0.430 | 0.781 | 0.658 | 0.832 | 0.548 | 0.730 |
| L3H2+L3H0 | 0.001 | 0.000 | 0.266 | 0.030 | 0.312 | 0.023 | 0.001 |
| control L3H3+L3H5 | 0.989 | 0.967 | 0.779 | 0.777 | 0.821 | 0.591 | 0.802 |
| L7H7 (late prev-token) | 0.967 | 0.832 | 0.824 | 0.774 | 0.665 | 0.586 | 0.869 |
- **One head (L3H2) carries the prev-token signal for the layer-4 induction heads** (layer-5 heads only partly: L5H6 0.82→0.51, L5H0 0.77→0.25). Removing it turns L4H6's induction attention from 0.97 to 0.06 and copying from 0.90 to 0.05. Removing two same-layer heads with low prev-token scores does not (the layer-4 heads even get sharper).
- The layer 8–9 heads partly survive without L3H2/L3H0 (L8H2 0.31). They're partly fed by the late prev-token head L7H7 (ablating it: L8H2 0.87→0.67). The two waves are wired partly separately, which fits their different formation times.
- Ablation = zeroing the head's output (z) before the output projection. Zero-ablation can be harsher than mean-ablation; I didn't test mean-ablation.

## 4. Late sweep: the second wave is gradual, and some first-wave heads fade (8 sequences per checkpoint)
| step | tokens | L4H6 | L4H10 | L4H11 | L4H8 | L8H2 | L9H5 | L9H8 | copy top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 2.1B | 0.93 | 0.97 | 0.69 | 0.34 | <0.03 | – | – | 0.508 |
| 8000 | 17B | 0.98 | 0.94 | 0.91 | 0.66 | 0.11 | – | – | 0.948 |
| 16000 | 34B | 0.98 | 0.94 | 0.84 | 0.56 | 0.28 | – | – | 0.946 |
| 32000 | 67B | 0.98 | 0.90 | 0.62 | 0.54 | 0.48 | – | – | 0.927 |
| 64000 | 134B | 0.95 | 0.85 | 0.31 | 0.38 | 0.59 | 0.49 | <0.23 | 0.928 |
| 143000 | 300B | 0.97 | 0.85 | 0.41* | 0.58 | 0.88 | <0.48 | 0.67 | 0.879 |
"–" = not in that checkpoint's top-12 list (the sweep printed only the top 12), so <12th-place score. *0.41 from the 8-sequence kcomp run; 16-seq main run: not in the top 12 (<0.48).
- **Two different dynamics.** Wave 1 (layers 4–5) is abrupt: 0 → 0.97 within one checkpoint interval (512→1000). Wave 2 (L8H2) is slow and steady over 140k steps: 0.11 → 0.28 → 0.48 → 0.59 → 0.88. My reading (one model): Olsson's abrupt "phase change" corresponds to wave 1.
- **Some wave-1 heads fade** (L4H11 0.91 → 0.31 by step 64k) while wave-2 heads grow. Head identity also drifts late (L9H5 0.49 at 64k, below the top list at the end; L9H8 appears). That fits Yin & Steinhardt's "slight decline" and their finding that many heads change role later in training. I haven't tested whether these heads become function-vector heads.
- Copying ability peaks early (≈0.95 at 8–16k) and slowly declines (0.88 at the end). Suggestive only (8 vs 16 sequences, one random-token test).

## 5. Olsson's ICL score on natural text (Pride & Prejudice, 16 windows × 11 tokens around each index): a SURPRISE, then a better test
| step | 0 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 32000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|
| median loss@50 − loss@500 | −0.06 | **1.35** | 0.99 | 0.40 | 0.50 | 0.81 | 0.64 | 0.55 | 0.29 |
| mean version | 0.00 | 0.41 | 0.26 | 0.28 | 0.12 | 0.23 | 0.22 | 0.21 | 0.06 |
- The score is already large at step 256, BEFORE any induction head exists. Olsson et al. report <0.15 nats before their phase change. Taken at face value this contradicts the co-occurrence argument for Pythia-160m.
- But my estimate is weak: the tokens at position 50 and position 500 are DIFFERENT tokens (content confound), there are only 16 windows, and the median and mean versions disagree by 3×. A step-256 model that has learned only unigram/bigram statistics could also gain from the context by "this text's word frequencies" (e.g. names repeating), which is a non-induction form of ICL.
- So I wrote a paired test (icl2.py): the same 62 target tokens, with their real distant context vs a foreign distant context (the last 50 tokens kept real), 24 windows, bootstrap CI. Running.
- Caveat for both: P&P is in the Pile, so "context helps" may partly be memorization cued by context.

### 5b. Paired test (icl2.py): same 62 targets, real vs foreign distant context, 24 windows, bootstrap 95% CI
| step | tokens | ICL benefit (nats) | 95% CI |
|---|---|---|---|
| 0 | 0 | 0.003 | −0.006, 0.013 |
| 256 | 0.54B | 0.004 | −0.006, 0.013 |
| 512 | 1.07B | 0.026 | 0.008, 0.046 |
| 1000 | 2.1B | 0.097 | 0.055, 0.142 |
| 2000 | 4.2B | 0.107 | 0.068, 0.147 |
| 8000 | 17B | 0.110 | 0.068, 0.154 |
| 143000 | 300B | 0.101 | 0.064, 0.139 |
- **The step-256 "ICL" of section 5 was the content confound.** With the same targets it's 0.004 (CI includes 0).
- **The shape matches Olsson's description (different estimator, 7 checkpoints):** ~0 → a jump in the same interval where the induction heads form (512→1000) → flat for the rest of training (0.10–0.11 from 2B to 300B tokens). The small positive value at 512 (0.026) fits the prev-token head already being partly formed.
- The size (0.10) isn't comparable to Olsson's 0.4: my measure only counts the benefit of tokens 50–450 back, with a P&P-specific text.
- Lesson for me: the "surprise" was a measurement artefact, and a matched design removed it. Before believing a result that contradicts a careful paper, fix the estimator first.

## 6. Does the circuit carry natural-text ICL? Ablation in the final model (paired test, 24 windows)
| zero-ablated | ICL benefit | 95% CI | (random-token copy top-1, from §1/§3) |
|---|---|---|---|
| none | 0.101 | 0.064, 0.139 | 0.88–0.90 |
| L3H2 (prev-token) | 0.088 | 0.051, 0.125 | 0.05 |
| L3H2+L3H0 | 0.084 | 0.047, 0.122 | 0.001 |
| control L3H3+L3H5 | 0.110 | 0.070, 0.152 | 0.80 |
| 6 wave-1 induction heads (L4H6/8/10/11, L5H0/6) | 0.076 | 0.043, 0.109 | 0.004 |
| 6 random low-score heads (L0H1, L1H5, L2H3, L6H4, L10H1, L11H9) | 0.108 | 0.063, 0.155 | – |
- **Dissociation.** Removing the wave-1 circuit wipes out exact copying of random sequences (0.90 → ~0), but the natural-text benefit of distant context falls only from 0.101 to 0.076 (6 heads, −25%), or to 0.084–0.088 (prev-token heads, −13 to −17%). The CIs overlap, so even that is suggestive, not significant.
- So in the trained Pythia-160m, most natural-text ICL (as I measure it) does NOT run through the circuit that defines "induction head". That is in the direction of Yin & Steinhardt 2025 (FV heads matter more for ICL; their induction-ablation ≈ random result is for models >1B, not 160m). It's a partial counterpoint to Olsson's hypothesis that "induction heads might constitute the mechanism for the actual majority of all in-context learning in large transformer models" (PRIMARY, verified on transformer-circuits.pub 10-23), which they rated strong only for small attention-only models.
- Not tested: the wave-2 heads (L8H2, L9H8) and other composition heads; other texts (P&P is probably memorizable: the Pile includes PG-19). Mean-ablation: see §8.
- Open (next): at step 1000, right after formation, does removing L3H2 kill the natural-text benefit? If yes, ICL starts out induction-based and diversifies later (wave 2, FV heads).

## 7. Same ablations at step 1000 (right after the phase change), paired test, 24 windows
| zero-ablated | step 1000 | 95% CI | final (step 143000) | 95% CI |
|---|---|---|---|---|
| none | 0.097 | 0.055, 0.142 | 0.101 | 0.064, 0.139 |
| L3H2 | 0.069 | 0.046, 0.092 | 0.088 | 0.051, 0.125 |
| L3H2+L3H0 | 0.044 | 0.026, 0.063 | 0.084 | 0.047, 0.122 |
| control L3H3+L3H5 | 0.097 | 0.054, 0.144 | 0.110 | 0.070, 0.152 |
| 6 wave-1 induction heads | **0.037** | 0.019, 0.057 | 0.076 | 0.043, 0.109 |
| 6 random low-score heads | 0.094 | 0.051, 0.141 | 0.108 | 0.063, 0.155 |
- **At step 1000 the induction circuit carries most of the natural-text ICL:** removing the 6 heads cuts it 0.097 → 0.037 (−62%), and the prev-token pair cuts it −55%. Both controls change nothing (0.094, 0.097).
- **At the end of training the same circuit carries only ~25%** (−25% for 6 heads, −17% for the pair), and the total benefit is the same (≈0.10).
- **Same output, different mechanism.** The ICL score stays flat from 2B to 300B tokens while its dependence on the wave-1 circuit falls from ~60% to ~25%. This suggests ICL starts out mostly induction-based and later relies mostly on other components (candidates: wave-2 induction heads, FV heads; not measured). The difference between the two fractions wasn't tested directly. (Correction 10-23: I first wrote that Olsson et al. 'flagged this possibility'. I searched their text and couldn't find that; removed.)
- Limits: one model, one text (memorizable), zero-ablation, 24 windows. The step-1000 six-head CI (0.019–0.057) just overlaps "none" (0.055–0.142); final-model differences are within the CIs. Direction is clear, size isn't.

## Summary (2026-10-23)
1. Prev-token head (L3H2) forms first; induction heads jump 0→0.97 between steps 512 and 1000 (1.1–2.1B tokens).
2. K-composition is causal: L3H2 alone gates the layer-4 induction heads.
3. Two waves: abrupt (layers 4–5), then slow (layers 8–9, 17B–300B tokens), with some wave-1 heads fading.
4. Natural-text ICL appears in the same interval and then stays flat (Olsson's shape), once the content confound is removed.
5. The mechanism under the flat score appears to shift: ~60% dependent on the wave-1 circuit at step 1000, ~25% at the end (suggestive: CIs overlap; difference not tested).

## 8. Robustness: mean-ablation (each head's output replaced by its mean over 8 reference P&P windows) gives the same picture
| ablated | step 1000 zero / mean | final zero / mean |
|---|---|---|
| L3H2+L3H0 | 0.044 / 0.043 | 0.084 / 0.084 |
| 6 wave-1 induction heads | 0.037 / 0.035 | 0.076 / 0.072 |
| 6 random heads | 0.094 / 0.097 | 0.108 / 0.101 |
(none: 0.097 at step 1000, 0.101 final.) So the mechanism shift in §7 isn't an artefact of zero-ablation. Caveat: the reference windows may overlap some target windows (same book).

## 9. Is the rest carried by the wave-2 induction heads? No (final model, paired test, 24 windows, zero-ablation)
| ablated | ICL benefit | 95% CI |
|---|---|---|
| none | 0.101 | 0.064, 0.139 |
| 6 wave-2 heads (L8H2, L8H8, L8H11, L9H6, L9H8, L9H9) | 0.098 | 0.064, 0.131 |
| all 12 induction heads (wave 1 + wave 2) | 0.075 | 0.042, 0.107 |
| 12 random low-score heads | 0.101 | 0.053, 0.152 |
| all 3 prev-token heads (L3H2, L3H0, L7H7) | 0.075 | 0.039, 0.109 |
- The wave-2 heads add nothing measurable on top of wave 1 (12 heads 0.075 ≈ 6 wave-1 heads 0.076). Removing every head with a high induction score, or every main prev-token head, leaves ~75% of the natural-text benefit. 12 random heads leave all of it.
- So in the final model, most of this benefit runs through heads that don't look like induction heads on the random-token test. Candidates (untested): function-vector-style heads, "semantic/topic" attention that isn't token matching, or context-cued memorization of P&P.

## 10. Layer scan: zero ALL 12 heads of one layer (final model, 24 windows; none = 0.101)
| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benefit | 0.117 | 0.103 | 0.102 | 0.087 | 0.104 | 0.108 | 0.110 | 0.101 | 0.104 | 0.106 | 0.091 | 0.106 |
- No single layer's attention is necessary for the natural-text benefit. Even zeroing all of layer 4 (which holds L4H6/L4H10/L4H11/L4H8) leaves 0.104.
- **This conflicts with §6** (6 heads from layers 4+5 → 0.076). Whole layer 4 → 0.104, whole layer 5 → 0.108. A superset ablation hurting less than a subset is possible (compensation, or the subset's damage elsewhere), but the simpler reading is that my unpaired, overlapping-CI comparisons are mostly noise. A paired comparison (same windows, per-window differences) is running (run12) to settle it. Until then, treat §6–§9 differences of ≤0.03 nats as unconfirmed.
- Consistent with Yin & Steinhardt (§4.2): below ~160M, neither induction- nor FV-head ablation changes token-loss ICL more than random.

## 11. Is the "6 heads hurt more than the whole layer" conflict real? Paired test on the saved per-window dumps (run13-paired-partial.txt)
- 6 wave-1 heads vs whole layer 4: −0.027 [95% CI −0.056, +0.003], 14/24 windows → **not significant** (the conflict with layer 4 may be noise).
- 6 wave-1 heads vs whole layer 5: −0.032 [CI −0.057, −0.006], 19/24 windows → **significant**: removing all of layer 5 hurts less than removing 4 L4 + 2 L5 heads.
- Candidate mechanisms from the literature (self-repair / backup-head papers, e.g. McGrath et al. 2023, Rushing & Nanda 2024): opposing heads within a layer, LayerNorm rescaling that differs by ablation size (zero-ablation is the worst case), anti-erasure. The discriminating experiments (complementary-subset ablations; mean/resample ablation; frozen LN) are the same kind of CPU runs the harness stopped on 10-24 for low memory. I'm holding them until Shalom says to restart ("תריץ").

## 12. Restarted runs (10-24, after the earlier runs were stopped): the puzzle and the shift, with paired tests (run16-paired-raw.txt, shift-test.txt)
### The whole-layer puzzle is explained by opposing heads (final model, paired vs none, 24 windows)
| zero-ablated | Δ benefit | 95% CI |
|---|---|---|
| 4 L4 induction heads (L4H6/8/10/11) | −0.010 | −0.024, +0.003 |
| the other 8 L4 heads | +0.012 | −0.017, +0.046 |
| whole L4 (§10) | +0.002 | −0.027, +0.030 |
| 2 L5 induction heads (L5H0/6) | −0.005 | −0.011, +0.003 |
| the other 10 L5 heads | +0.013 | −0.008, +0.033 |
| whole L5 (§10) | +0.007 | −0.014, +0.028 |
| the 6 together (w1) | **−0.025** | −0.041, −0.009 |
- Whole layer ≈ induction part + rest part: L4 −0.010 + 0.012 = +0.002 (observed +0.002); L5 −0.005 + 0.013 = +0.008 (observed +0.007). The non-induction heads of layers 4–5 slightly *hurt* distant-context use, so removing a whole layer cancels out. Each part alone is n.s.; the 6 induction heads together are significant (mildly superadditive: −0.025 vs −0.015 summed).
### The mechanism shift, tested properly
- Step 1000: w1 −0.0605 [−0.106, −0.014] (18/24 windows); random 6 heads −0.003 [−0.013, +0.006].
- Share of the benefit removed by w1 (joint bootstrap over the same 24 windows): step 1000 **0.63 [0.24, 0.81]**, final **0.25 [0.10, 0.40]**, difference **0.37 [−0.00, 0.57]**.
- So the direction holds and the step-1000 drop is 2.4× the final one with both CIs excluding 0, but the difference in *shares* is only borderline (lower bound ≈ 0). Fair statement: "the induction circuit's share of the natural-text benefit falls from roughly 60% to roughly 25%; the decline is at the edge of significance with 24 windows."
- L3H2 alone: step 1000 −0.029 [−0.057, +0.001] (−29%), final −0.014 [−0.033, +0.009] (−13%). Same direction as the 6-head result, both borderline/n.s.

## 13. Are the "helpful-to-remove" L4/L5 heads copy-suppression heads? (copy_scores.py → run17; OV copy test after Wang et al. 2022 / McDougall et al. 2023)
- Score per head: over 1,000 tokens, how often the head's direct OV path (effective embedding → OV → unembedding) puts the token's OWN logit in the top 5% (copying) or the bottom 5% (suppressing). Chance = 5% each.
- **Strong suppressors exist, but late:** L8H9 0.89, L9H5 0.86, L10H2 0.83 (then L7H2 0.42, L6H7 0.33). That's the same depth pattern McDougall et al. report for GPT-2 small (L10H7, L11H10), here in Pythia-160m. (L8H9 is also the ordinal/letter successor head in my separate number experiments (github.com/NitzanBrain/clocks-in-the-weights), so its "suppression" may partly be "predict the next item, not this one".)
- **The non-induction heads of layers 4–5 are NOT suppressors** (bottom-5% share ≤ 0.14; most ≈ chance). The strong copiers there are exactly the induction heads (L4H10 0.94, L5H0 0.96, L5H6 0.89, L4H6 0.78) plus L5H7/L5H9 (0.79/0.80).
- So copy suppression does NOT explain why removing the other L4/L5 heads slightly helps (§12; those effects were n.s. anyway). Remaining candidates: zero-ablation side effects (norm/LN shifts), or a different interaction. Not resolved.
