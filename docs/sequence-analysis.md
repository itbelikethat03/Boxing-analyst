# Sequence analysis (as built, M2)

Code: `src/boxing_ai/sequences/` (primitives) and `src/boxing_ai/analytics/` (questions). Everything is a pure
function of a list of `Event`s — no database, no UI — and fully deterministic.

## Pipeline

```
events ─ filter categories ─ order ─ segment into bursts ─ tokenize ─ n-grams / transition tables
```

- **Categories** are filtered *first* (default: PUNCH + FEINT + DEFENSE; footwork excluded), so gaps are measured between
  the events that remain.
- **Tokenizer** = projection `Event -> str`: `by_action` (`SLIP`), `by_action_direction` (`SLIP_LEFT`),
  `by_action_target` (`HOOK_BODY`), or `actor_tagged(...)` for two-fighter streams (`SELF:JAB`, `OPP:SLIP`).
- **Bursts** are derived views, never stored. A new burst starts when, between consecutive events:
  - the silence since the burst's latest end exceeds `gap_ms` (`gap == gap_ms` stays together; overlaps count as 0),
  - the video or round changes, or
  - an unobserved interval (replay/cutaway/unusable angle) lies between them.
  Defaults are starting guesses to tune on real data: `DEFAULT_COMBO_GAP_MS = 1000` (one fighter's combination),
  `DEFAULT_EXCHANGE_GAP_MS = 2500` (exchange between both fighters).
- **n-grams never span bursts.** Counts are ranked by count descending, ties by token order.
- **Transition tables** pad each burst `START … END`, so `P(JAB|START)` is an entry pattern, `P(END|CROSS)` an exit
  pattern, and each context's probabilities sum to 1. Every `Transition` carries `count` and `total`: small samples
  are the norm, so always show them next to the probability (`min_count` hides rows but keeps true totals).

## Questions answered (`analytics`)
| Question | Function |
|---|---|
| What are the most common combinations? | `top_ngrams(events, n, spec)` |
| P(next \| previous) / what follows X? | `transition_probabilities`, `next_after` |
| How do bursts start / end? | `entry_patterns`, `exit_patterns` |
| What does he throw, and how much per round (per *observed* minute)? | `action_distribution`, `round_stats` |
| How often does each punch land / get blocked / miss? | `outcome_breakdown` (landed% over *known* outcomes) |
| Which sequences pay off? | `pattern_outcomes(events, n)` — `top_ngrams` plus how often the sequence's **final** punch landed |
| How does he react to the opponent's jab? | `next_after(both fighters' events, "OPP:JAB", spec_with_actor_tagged_tokenizer)` |
| Slice by fighter/fight/video/round/time | `select_events` (SQL does this for real data from M4) |

## Hand-calculated golden example
Stream (one fighter, one burst): `JAB JAB CROSS SLIP_LEFT JAB CROSS` — fixture in `tests/fixtures/golden.py`.

- Unigrams: JAB 3, CROSS 2, SLIP_LEFT 1
- Bigrams (5 = n−1): JAB→JAB 1, **JAB→CROSS 2**, CROSS→SLIP_LEFT 1, SLIP_LEFT→JAB 1
- Trigrams (4): each of JAB→JAB→CROSS, JAB→CROSS→SLIP_LEFT, CROSS→SLIP_LEFT→JAB, SLIP_LEFT→JAB→CROSS once
- With sentinels: P(JAB\|START)=1 · P(JAB\|JAB)=1/3 · **P(CROSS\|JAB)=2/3** · P(SLIP_LEFT\|CROSS)=1/2 ·
  **P(END\|CROSS)=1/2** · P(JAB\|SLIP_LEFT)=1
- *Why END matters:* without it P(SLIP_LEFT\|CROSS) would read 1/1 = 100 %, silently ignoring the cross that ended the burst.
- Add a lone JAB 30 s later → a second burst: bigrams unchanged (no CROSS→JAB across the pause), and
  P(CROSS\|JAB) drops to 2/4 = 1/2 because that jab was followed by END.

## Verification performed
- All numbers above are asserted in `tests/unit/test_analytics_patterns.py` / `test_transitions.py`.
- Mutation-checked: deliberately breaking the gap comparison, the END sentinel, the ranking tie-break, and the
  burst-end logic each makes tests fail (then reverted).
- M4 will add an independent cross-check of the same fixture against a SQL `LEAD()` implementation.
