# SDD ledger — plan: docs/superpowers/plans/2026-10-16-review-remediation.md

Spec: notes/PROJECT_REVIEW_2026-10.md. Branch start: 223819f (main 2497820 + docs commit).
User request: subagent-driven, then loop review→debug→fix; 3 loops total; return best result. Loop 1 = SDD over all tasks + final review; loops 2-3 = whole-branch audit (opus reviewer + Codex + dual-precision/slow/probe runs) → one fix wave → scoped re-review. Each loop head tagged loop-1/2/3 (local tags only).

## Pre-flight scan
| Pair / task | Shared | Finding | Ruling |
|---|---|---|---|
| 1.1 ↔ 1.2 ↔ 3.3 | engine signature | 1.1 interface lists final args order; 1.2 appends max_attempts, 3.3 adds static step_update | Ruling: each task extends the signature in order; callers updated in the task that changes it — why: plan says so — cost if wrong: extra call-site churn |
| 0.2 ↔ 2.1 | solvers/implicit.py | 0.2 changes tol defaults, 2.1 changes initial done | sequential, no conflict |
| 2.2 ↔ 3.4 | PD/NS trivial-mode exclusion | 2.2 runs before trivial_index exists | Ruling: 2.2 keeps argmin|μ−1| exclusion in one helper (`_trivial_index`) that 3.4 replaces — why: ordering — cost: small rework |
| 2.x ↔ 3.1/3.3 | detect_events rhs signature | PD/NS need user args; refine needs step_update args | Ruling: detect_events gains keyword `args=None, step_update=None`; orchestrator builds the 2-arg rhs per bracket from f/args (step_update applied at left point); PD/NS keep raw_f/mesh fields and gain `args: PyTree = None` like codim-2 events — why: Event objects never see problem.args — cost: API field addition |
| 3.1 ↔ api.py:398 | problem.args tuple unpack | becomes attribute access | in 3.1 |
| 4.1 | ContinuationPar.loop | plan lists scan_continuation.py; field lives in api.py | Ruling: field in api.py, engine variant in scan_continuation.py |
| 5.3 | ContinuationSolution | used internally for plot/save | Ruling: keep ContinuationSolution (internal + export unchanged); remove ContinuationProblem only if grep shows no internal use — cost: one legacy class lingers |
| 5.1 | arclength weighting | changes step counts of existing tests | Ruling: adjust numeric expectations in existing tests only where they encode step counts, with comment — cost: test churn |
| 2.2 step 5 | refers to Phase 6.4 (not a task) | informational | no action |
| all tasks | self-consistency | tests named in each task match the code each task specifies; files created are the ones later touched | checked: consistent |

## Progress
Task 0.1: complete (commits 223819f..ef22d4d, review clean)
Task 0.1: minor (deferred): test_is_x64_matches_default_float tautological; branch_switch seed bound loosened only checks seed point; imports inside test fn
Task 0.2: fix round 1/5 (2 addressed, 1 open — collocation_tol blanket 1e-5 floor; commits e102f08..273f859)
Task 0.2: fix round 2/5 (implementer: collocation_tol removed, explicit tol/corrector_tol overrides; commits 273f859..157077b) — scoped RE-REVIEW of round 2 STILL PENDING (findings: blanket collocation_tol floor; diff 273f859..157077b)
Note: slow test test_adaptx_period_doubling_switch_matches_matcont fails on untouched baseline too (RuntimeError "found no adaptx PD crossing", pd_switch.py:96) — pre-existing, not caused by this branch; track in loop audit.

## HANDOFF (2026-10-16, Mac → Ubuntu)
- Paths in these files use `<repo>` = repository root of this branch's checkout; Python = `<repo>/.venv/bin/python` (create with `uv venv && uv pip install -e ".[dev]"`).
- review-*.diff files are not committed (regenerate with the skill's scripts/review-package PLAN BASE HEAD).
- Resume point: (1) scoped re-review of Task 0.2 round 2 (FIX_BASE 273f859, HEAD 157077b) using rereview-instructions.md; (2) then Task 1.1 onward. Tasks 0.1 complete; 0.2 pending re-review.
- Loop structure the user asked for: Loop 1 = SDD over all tasks + final whole-branch review (opus) + one fix wave; then `git tag loop-1`. Loops 2 and 3 = whole-branch audit (opus reviewer + optional Codex read-only review + full suite under X64=0/1 + slow suite + the review's original probes) → ONE fix subagent → scoped re-review → `git tag loop-2` / `loop-3`. Finally compare loops (tests passing, findings open, probe outcomes) and report the best one. Tags are local unless the user says to push them.
- Session conventions: reply to the user in Persian (their CLAUDE.md); repo docs/code stay English.
