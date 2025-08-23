# Conversation Quality Evaluation (0–100)

Score across **10 dimensions** (each 0–100), then apply weights (sum = 100) to produce a final score. Every dimension includes: **what to measure**, **how to score**, and **penalties/bonuses**.

### Weights

1. Thread Weaving & Long-Horizon Relevance — **15**
2. Depth & Insightfulness — **12**
3. Curiosity & Timed Exploration — **12**
4. Investigative Challenge & Perspective — **10**
5. Personal & Emotional Connection — **10**
6. Knowledge-Aware Probing (Guest-Specific) — **10**
7. Engagement & Rhythm (incl. Smart Micro-Prompts) — **9**
8. Accessibility & Clarity — **8**
9. Balance & Turn-Taking — **7**
10. Memorability & Impact — **7**

---

## 1) Thread Weaving & Long-Horizon Relevance (15)

**Goal:** Reward follow-ups that connect not only to the **last turn** but to **distant prior turns**; deeper, less-obvious yet relevant links score higher.

**Signals to extract**

* **Callback depth**: how many turns back the referenced content originated (d ≥ 1).
* **Semantic relevance** of the link (cosine similarity between current question and referenced earlier span).
* **Obscurity/Non-obviousness**: inverse popularity of the linked concept across the convo (rarer → higher score).
* **Bridge quality**: explicit connective language (“earlier you said…”, “how does X relate to Y from before?”).

**Scoring (0–100)**

```
callback_score = mean over callbacks of [ sim * g(d) * obscurity ]
where:
  sim ∈ [0,1] (semantic similarity)
  g(d) = min(1.0, 0.35 + 0.25*log2(d+1))   // deeper callbacks get more credit; capped
  obscurity ∈ [0.7,1.15] based on how infrequently the referenced concept appears
thread_weaving = 100 * clamp01(weighted_avg(callback_score) - repetition_penalty)
```

**Penalties/Bonuses**

* **Penalty**: redundant callbacks to the *same* prior point without adding new angle (−5 to −15).
* **Bonus**: multi-hop connections (A→B from turn 6 + C from turn 19) that resolve into a coherent synthesis (+5 to +12).

**Rubric bands**

* 0–20: Only superficial, immediate follow-ups; no distant connections.
* 21–60: Some callbacks; mostly near-term or obvious.
* 61–85: Regular, meaningful long-horizon weaving.
* 86–100: Multiple deep, non-obvious, highly relevant callbacks that advance insight.

---

## 2) Depth & Insightfulness (12)

**Signals**

* Ratio of **probing questions** (“why/how/what if/so what/implications”) to fact-retrieval.
* **Layering**: follow-ups that move from data → meaning → consequence → trade-offs.
* **Concept density**: presence of abstractions (assumptions, models, mechanisms).

**Scoring**

```
depth = 100 * (0.5*probing_ratio + 0.3*layering_index + 0.2*concept_density)
```

**Bands**

* 0–20 shallow | 21–60 mixed | 61–85 strong probing | 86–100 consistently profound.

---

## 3) Curiosity & Timed Exploration (12)

**Goal:** Measure *new knowledge shared* and *new angles explored* **at the right time** (not mechanical).

**Signals**

* **Novelty**: information units from guest not previously present (unique entities/claims).
* **Angle shifts**: topic transitions that remain semantically tied to the core thread.
* **Timing fitness**: exploration follows a completed thought or natural beat (not mid-answer).

**Scoring**

```
curiosity = 100 * (0.4*novelty_rate + 0.35*relevant_angle_shifts + 0.25*timing_fitness)
```

**Penalty**: detours that derail coherence (−10 to −25).
**Bonus**: elegant pivot lines (“what if we flip that assumption…”) (+5 to +10).

---

## 4) Investigative Challenge & Perspective (10)

**Goal:** Respectfully **challenge**, cross-examine, introduce **doubt** and **new perspectives**.

**Signals**

* **Challenge instances**: targeted, evidence-seeking follow-ups.
* **Perspective shifts**: counterfactuals/comparators (“how would this look if…?”).
* **Tone safety**: challenge without hostility (sentiment/politeness markers).

**Scoring**

```
challenge = 100 * (0.45*challenge_density + 0.35*perspective_shifts + 0.20*tone_safety)
```

**Penalty**: adversarial or badgering tone (−15 to −30).

---

## 5) Personal & Emotional Connection (10)

**Goal:** Did the interviewer touch something **deep/personal** and handle it with care?

**Signals**

* **Personal disclosure prompts** (moments, feelings, values).
* **Emotional resonance**: detected emotion in guest responses; pauses acknowledged.
* **Continuity**: returning to a personal thread later with care.

**Scoring**

```
personal = 100 * (0.4*personal_prompts + 0.4*guest_emotional_depth + 0.2*continuity)
```

**Bonus**: skillful shepherding through a vulnerable moment (+8 to +12).

---

## 6) Knowledge-Aware Probing (Guest-Specific) (10)

**Goal:** Use **known facts** about the guest intelligently to probe deeper (not trivia-quizzing).

**Signals**

* **Fact grounding**: references to guest’s works, history, stated positions.
* **Insight leverage**: the reference is used to **advance** a line of inquiry.
* **Accuracy**: zero factual misattributions.

**Scoring**

```
knowledge_probe = 100 * clamp01( 0.5*grounded_refs + 0.4*leverage_quality + 0.1*accuracy )
```

**Penalty**: factual miss (−25 each, floor at 0 for this dimension).

---

## 7) Engagement & Rhythm (incl. Smart Micro-Prompts) (9)

**Goal:** Conversational **energy** with skilled use of **micro-prompts** and brief closed questions to propel, not derail.

**Signals**

* **Rhythm**: variance in pace; strategic pauses acknowledged.
* **Micro-prompt quality**: short turns (<4 tokens) like “go on?”, “why?”, “and then?” used **sparingly** and **timed** after dense content.
* **Micro-prompt rate & spacing**: ideal ≤10% of interviewer turns, spaced ≥3 turns apart on average.
* **Humor/play markers** where appropriate.

**Scoring**

```
micro_rate_score = 1 - abs(actual_rate - 0.07)/0.07   // peaks near ~7%
micro_spacing = min(1, avg_gap/3)                     // ≥3-turn spacing ideal
micro_timing = fraction used after dense/peak moments
engagement = 100 * (0.35*rhythm_variance + 0.25*micro_rate_score + 0.2*micro_spacing + 0.2*micro_timing)
```

**Penalty**: machine-gun “uh-huh/okay” fillers (−10 to −25).
**Bonus**: a well-timed, tiny closer (“fair.” “hm.”) that unlocks a long, rich answer (+5).

---

## 8) Accessibility & Clarity (8)

**Signals**

* **Readability** of questions (Flesch-Kincaid or similar).
* **Jargon ratio** (unexplained technical terms).
* **Analogy count** tied to clarity (not fluff).

**Scoring**

```
access = 100 * (0.4*readability_norm + 0.35*(1 - jargon_ratio) + 0.25*effective_analogies)
```

---

## 9) Balance & Turn-Taking (7)

**Signals**

* **Word share**: interviewer \~30–40%, guest \~60–70%.
* **Interruptions**: minimal and purposeful.
* **Turn ratio**: not ping-pong dominance.

**Scoring**

```
balance = 100 * (0.6*share_closeness + 0.25*turn_ratio_fairness + 0.15*(1 - interruption_rate))
```

---

## 10) Memorability & Impact (7)

**Signals**

* **Quotable lines** (semantic distinctiveness).
* **A-ha moments** (conceptual shift markers).
* **Closing lift** (reflection/future-facing synthesis).

**Scoring**

```
memorability = 100 * (0.45*distinctive_phrases + 0.35*aha_density + 0.20*closing_quality)
```

---

# Final Aggregation

**Final Score (0–100) =**

```
0.15*thread_weaving
+0.12*depth
+0.12*curiosity
+0.10*challenge
+0.10*personal
+0.10*knowledge_probe
+0.09*engagement
+0.08*access
+0.07*balance
+0.07*memorability
```

---

## Operational Rules (for the Evaluator Agent)

1. **Evidence tagging**: For each subscore, cite 1–3 concrete spans (turn indices or timestamps).
2. **Normalization**: Compute per-dimension raw scores, then clamp to \[0,100].
3. **False-positive guards**:

   * Do not count name-drops as knowledge-aware probing unless the reference **influences** the next question.
   * Curiosity/timing credit only if the pivot follows a natural boundary (guest finishes a thought; pause detected).
4. **Redundancy filter**: If two callbacks target the **same** prior claim without adding a new angle, apply redundancy penalty once per segment.
5. **Tone safety**: If sentiment analysis shows sustained negativity without mitigating politeness markers, cap **Challenge** at 65.
6. **Filler sanity**: If micro-prompts exceed 15% of interviewer turns **or** cluster ≤2 turns apart for >3 instances, apply a stacking penalty to **Engagement**.

---

## Minimal Scorecard Template (per conversation)

**Meta**

* Title / Guest / Date / Duration
* Evaluator / Model version
* Transcript source

**Dimension Scores (0–100) with Evidence**

1. Thread Weaving & Long-Horizon Relevance: \_\_ /100

   * Evidence: \[turn 42→7], \[turn 58→19], notes…
2. Depth & Insightfulness: \_\_ /100

   * Evidence: …
3. Curiosity & Timed Exploration: \_\_ /100

   * Evidence: …
4. Investigative Challenge & Perspective: \_\_ /100

   * Evidence: …
5. Personal & Emotional Connection: \_\_ /100

   * Evidence: …
6. Knowledge-Aware Probing: \_\_ /100

   * Evidence: …
7. Engagement & Rhythm: \_\_ /100

   * Evidence: …
8. Accessibility & Clarity: \_\_ /100

   * Evidence: …
9. Balance & Turn-Taking: \_\_ /100

   * Evidence: …
10. Memorability & Impact: \_\_ /100

* Evidence: …

**Weighted Final Score:** \_\_\_\_ /100
**Top 3 Strengths:** …
**Top 3 Improvements:** …
**Notable Quotes:** …

---

## Implementation Hints (so it’s buildable)

* Use embeddings (e.g., cosine similarity) to compute **sim** for callbacks and angle shifts.
* Detect **callback depth** by aligning references (coreference/entity linking) to earlier turns and measuring turn distance.
* **Obscurity**: 1 / (1 + frequency of the referenced concept across turns).
* **Novelty rate**: unique entity/claim extraction vs prior context window.
* **Timing fitness**: pause/turn-end detection + completion markers; avoid mid-sentence pivots.
* **Micro-prompt detection**: interviewer turns with ≤4 tokens and interrogative or continuer patterns (“go on”, “and then?”, “why?”).
* **Politeness/tone**: combine sentiment with hedges (“could we”, “help me understand”), honorifics, and mitigations.