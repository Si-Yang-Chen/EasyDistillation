---
name: easydistillation-git-push
description: >-
  Manages EasyDistillation dual-remote git workflow: daily pushes go to the
  private fork; IHEP public origin is updated only after tests pass. Use when
  committing, pushing, syncing branches, or when the user mentions git push,
  private repo, or IHEP public repository for this project.
---

# EasyDistillation Dual-Remote Push

## Remote layout

| Remote | URL | Role |
|--------|-----|------|
| `private` | `git@github.com:Si-Yang-Chen/EasyDistillation.git` | **Default push target** — WIP, experiments, pre-review |
| `origin` | `git@github.com:IHEP-LQCD/EasyDistillation.git` | **Public IHEP repo** — only after tests pass |

`master` is configured with `branch.master.pushRemote = private`, so plain `git push` targets `private`.

## Policy

以后都是在测试通过后再提交到IHEP的公共仓库。

- Push to `private` freely during development.
- Push to `origin` (IHEP public) **only after** the CI-equivalent test suite passes locally.
- Never push to `origin` when tests fail or were skipped without user approval.
- Never force-push to `origin/master` unless the user explicitly requests it.

## Standard workflow

### 1. Commit (local)

Follow the repo commit rules: stage only relevant files, concise message, no `--no-verify` unless requested.

### 2. Push to private (default)

```bash
git push -u private HEAD
```

Or, on a configured branch such as `master`:

```bash
git push
```

### 3. Gate before public push

Run the same CPU tests as CI:

```bash
pip install -e ".[dev]"   # if not already installed
pytest -m "not (gpu or mpi or integration)" -v
```

Optional lint (CI also runs this):

```bash
ruff check lattice/
```

**Proceed to public push only if pytest exits 0.** If tests fail, fix issues or push to `private` only.

### 4. Push to IHEP public (after tests pass)

```bash
git push origin HEAD
```

For `master`:

```bash
git push origin master
```

## Agent checklist

When the user asks to commit and/or push:

```
- [ ] Confirm which changes belong in the commit (exclude unrelated files)
- [ ] Create commit if requested
- [ ] Push to private (default)
- [ ] Run pytest -m "not (gpu or mpi or integration)" -v
- [ ] If tests pass AND user wants public sync → git push origin <branch>
- [ ] If tests fail → report failures; do not push to origin
```

If the user only says "push" without mentioning public/IHEP/origin, push to `private` only.

If the user explicitly asks to push to IHEP/public/origin, run tests first; abort public push on failure unless the user overrides after seeing results.

## Common commands

```bash
# Remotes
git remote -v

# Fetch both
git fetch private
git fetch origin

# Push current branch to private (default)
git push

# Push current branch to IHEP public (after tests)
git push origin HEAD

# Set upstream on first push to private
git push -u private <branch>
```

## Troubleshooting

**`private` remote missing**

```bash
git remote add private git@github.com:Si-Yang-Chen/EasyDistillation.git
git config branch.master.pushRemote private
```

**Plain `git push` goes to origin instead of private**

```bash
git config branch.<branch>.pushRemote private
```

**SSH auth errors** — user must have deploy key or SSH key authorized for `Si-Yang-Chen/EasyDistillation` and `IHEP-LQCD/EasyDistillation`.
