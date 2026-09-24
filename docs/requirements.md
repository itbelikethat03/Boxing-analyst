# Requirements

## Goal
Accept boxing footage and (eventually) automatically produce a structured timeline of boxing actions,
store them as events, and analyse the sequences a fighter uses.

## Fundamental principle
**Individual actions are the fundamental data representation.** Combinations such as `JAB_JAB_CROSS`
are derived from event streams at analysis time and are never stored types or ML classes.

## MVP
> Given manually annotated broadcast footage (CSV + TOML), import individual actions into PostgreSQL and
> generate deterministic statistical analyses of boxing sequences, viewable in a basic interface.

Out of scope for the MVP: any computer vision/ML, FastAPI, a video-annotation UI, multi-user auth,
sophisticated sequence mining.

## Footage
Broadcast professional fights (camera cuts, replays, multiple angles) — see `docs/plan.md` §0/§5 for why
this forces an explicit notion of *unobserved intervals*.

## Questions the system must eventually answer
- What punches does this fighter throw most? What does he throw after his jab?
- Which combinations does he use most? Which defensive actions follow his combinations?
- Which sequences are specific to him? How do patterns change between rounds/fights?
- Can I click a combination and watch the corresponding section of video?

See `docs/plan.md` for milestones, risks, and open assumptions.
