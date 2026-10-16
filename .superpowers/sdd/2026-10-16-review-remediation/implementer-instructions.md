# Implementer instructions (shared by every task)

You implement ONE task of the JaxCont 2026-10 remediation plan (a JAX numerical continuation/bifurcation library).

1. Read your task brief (path given in your dispatch) — it is your requirements, with exact values to use verbatim.
2. Read global constraints + environment: <repo>/.superpowers/sdd/2026-10-16-review-remediation/global-constraints.md
3. If a requirement is unclear, ask (reply NEEDS_CONTEXT) instead of guessing.
4. Implement exactly what the task specifies with TDD (write failing test, see it fail, implement, see it pass). Follow existing code patterns; don't restructure outside the task.
5. While iterating run focused tests; before committing run the full default suite under BOTH `JAXCONT_TEST_X64=0` and `JAXCONT_TEST_X64=1` (`$PY -m pytest -q -n 8 -p no:cacheprovider`). Where the brief touches MatCont/periodic/codim-2 numerics also run `-m slow` once (float32 is fine for slow).
6. Commit in the worktree (messages end with the Co-Authored-By line). Never push, never bare git stash, stay inside <repo>.
7. Self-review your own diff (completeness, names, YAGNI, tests verify behavior, pristine output). Fix what you find.
8. You never dispatch subagents or reviewers. Review comes from the controller.
9. Escalate (BLOCKED/NEEDS_CONTEXT) when the task needs an architectural choice the brief doesn't settle.

Report: write the full report (what you implemented, tests + results, TDD RED/GREEN evidence with commands and output excerpts, files changed, self-review findings, concerns) to the report path in your dispatch. If later resumed with review findings: fix, re-run covering tests, APPEND a fix report (changes, tests, command, output) to the same file.
Reply with ONLY (<15 lines): Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary (both precisions), concerns, report path.
