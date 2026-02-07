## Summary

<!-- One paragraph: what this PR changes and why. -->

## What Changed

<!-- Bullet the user-facing and architectural deltas. -->

## Validation

<!-- Prefer exact commands and paste key output if non-trivial. -->

- [ ] `mise run install-test-deps`
- [ ] `mise run test-unit`
- [ ] `mise run test-integration`

Optional (only if relevant to this PR):

- [ ] `python explorer.py --list-transforms`
- [ ] `python explorer.py validate`
- [ ] `python explorer.py run --transforms default`
- [ ] `mise run docs-generate && mise run docs-verify`
- [ ] `mise run assets-regenerate` (only when render inputs changed)

## Notes for Review

<!-- Call out risk, tradeoffs, follow-ups, and anything reviewers should focus on. -->

## Checklist

- [ ] Scope is tight; unrelated changes excluded
- [ ] Docs updated (if behavior changed)
- [ ] Tests added/updated (if logic changed)
- [ ] Generated references/assets are committed where required
