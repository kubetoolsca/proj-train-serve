# Contributing Guidelines

---

## 1. Branching Model

This repository uses:

```text
develop -> integration branch
main    -> stable release branch
```

Rules:

- Create all normal work branches from `develop`.
- Open pull requests into `develop`.
- `main` should only receive release PRs from `develop`.
- Do not commit directly to `main`.
- Do not commit directly to `develop` unless explicitly approved.

---

## 2. Branch Naming

Every branch must include the GitHub issue number.

Allowed prefixes:

```text
feat/
bug/
fix/
maint/
docs/
test/
ci/
research/
perf/
```

Format:

```text
<prefix>/<issue-number>-short-description
```

Examples:

```text
feat/12-fashion-mnist-training
feat/26-cifar10-support
bug/18-fix-onnx-export
maint/1-repo-foundation
docs/7-update-argo-docs
ci/20-self-hosted-runners
```

---

## 3. Pull Request Naming

Every PR title must include the issue number.

Format:

```text
[#<issue-number>] <type>: short description
```

Examples:

```text
[#1] maint: add repository foundation
[#12] feat: add Fashion-MNIST training
[#26] feat: add CIFAR-10 support
[#20] ci: configure self-hosted runners
```

---

## 4. Commit Message Style

Preferred format:

```text
<type>(<area>): short message
```

Examples:

```text
feat(training): add Fashion-MNIST data module
feat(data): add CIFAR-10 labels
test(api): add preprocessing tests
docs(readme): update directory structure
ci(github): add self-hosted runner workflow
```

Allowed types:

```text
feat
fix
bug
docs
test
ci
maint
refactor
research
perf
```

---

## 5. Required PR Checklist

Every PR must satisfy:

```text
[ ] My branch name includes the GitHub issue number.
[ ] My PR title includes the GitHub issue number.
[ ] My PR targets develop, unless this is a release PR.
[ ] I added or updated tests where required.
[ ] I ran tests locally.
[ ] I updated README.md if commands, behavior, setup, or structure changed.
[ ] I updated the README directory structure if files/folders changed.
[ ] I updated CHANGELOG.md for user-visible changes.
[ ] I updated docs/ if architecture, workflow, serving, or training behavior changed.
[ ] CI passes.
```

---

## 6. README Update Rule

Update `README.md` in every PR that changes:

- installation steps
- run commands
- training behavior
- serving behavior
- Docker usage
- Kubernetes manifests
- Argo Workflows
- Argo Events
- directory structure
- environment variables
- CI behavior
- project architecture

The README directory structure must always match the repository.

---

## 7. Changelog Rule

Maintain `CHANGELOG.md`.

Update it for:

- new features
- bug fixes
- breaking changes
- documentation changes that affect usage
- CI changes
- training pipeline changes
- serving changes
- release notes

Use:

```markdown
## [Unreleased]

### Added

### Changed

### Fixed

### Removed
```

---

## 8. Dependency Management

This project uses:

```text
pyproject.toml
uv
```

Install dependencies:

```bash
uv sync --dev
```

Run commands using:

```bash
uv run <command>
```

Examples:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

---

## 9. Testing Requirements

Add or update tests for all non-documentation changes.

Expected test areas:

### Training

```text
data loading
model forward pass
training step
checkpoint creation
```

### ONNX

```text
ONNX export
ONNX Runtime model load
PyTorch vs ONNX output comparison
```

### API

```text
image preprocessing
logits postprocessing
FastAPI health endpoint
FastAPI prediction endpoint
```

### CI

```text
workflow syntax
linting
tests
manifest checks
```

---

## 10. Documentation Requirements

Documentation must be beginner-friendly.

When adding a tool or concept, explain:

```text
What is it?
Why do we use it?
Where is it used in this project?
How can I run it?
How can I verify it worked?
```

---

## 11. Release Flow

Only maintainers should release.

Release process:

```text
1. Merge completed work into develop.
2. Confirm CI passes on develop.
3. Update CHANGELOG.md.
4. Open release PR from develop to main.
5. Merge after review.
6. Tag main.
7. Create GitHub Release.
```

Tag format:

```text
vX.Y.Z
```

Example:

```bash
git checkout main
git pull origin main
git tag v0.1.0
git push origin v0.1.0
```

---

## 12. Versioning

Use semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Examples:

```text
0.1.0 repository foundation
0.2.0 Fashion-MNIST local training
0.3.0 CIFAR-10 support
0.4.0 Argo pipeline
0.5.0 Triton and FastAPI serving
1.0.0 complete end-to-end demo
```

---

## 13. Skipping CI

GitHub Actions supports skipping `push` and `pull_request` workflows when the commit message contains one of these tokens:

```text
[skip ci]
[ci skip]
[no ci]
[skip actions]
[actions skip]
```