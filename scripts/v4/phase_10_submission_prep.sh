#!/usr/bin/env bash
# Phase 10: submission preparation — bioRxiv, GitHub release, PyPI check.
# Actual submission requires user account / web upload, so this script
# stages everything and emits READY-FOR-USER-ACTION instructions.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

mkdir -p "$ROOT/submissions/biorxiv" "$ROOT/submissions/release"

# Check PyPI package name availability
python3 - <<'PY' || true
import urllib.request
import json
for name in ("perturbcausal", "causalcellbench", "perturb-causal"):
    try:
        u = urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=5)
        print(f"  {name}: TAKEN on PyPI")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  {name}: AVAILABLE on PyPI")
        else:
            print(f"  {name}: check failed ({e})")
    except Exception as e:
        print(f"  {name}: check failed ({e})")
PY

# Generate USER_ACTIONS.md with exact steps the user must take
cat > "$ROOT/submissions/USER_ACTIONS.md" <<'MD'
# Manual submission steps (required after orchestrator completes Phase 10)

These steps require account credentials / web uploads and cannot be automated.

## 1. bioRxiv v0 (priority anchor) — post immediately after Phase 1

Upload `submissions/biorxiv/PerturbCausal_v0.pdf` with:
- Subject: Genomics + Systems Biology
- License: CC-BY 4.0
- Cover letter: `submissions/biorxiv/cover_letter.md`
- Authors: same as your usual bioRxiv profile

After submission, save the DOI to `submissions/biorxiv/v0_doi.txt`.

## 2. GitHub release v1.0.0

```
cd "/Users/aayanalwani/virtual cells"
git tag -a v1.0.0 -m "PerturbCausal v1.0.0 — first stable release"
git push --tags
gh release create v1.0.0 --notes-file submissions/release/release_notes.md
```

## 3. PyPI publish

```
python3 -m build
twine upload dist/*
```

(Requires PyPI API token in ~/.pypirc.)

## 4. TMLR submission

Submit via OpenReview. Use the bioRxiv link as priority anchor.
MD

echo "  Phase 10: USER_ACTIONS.md written. See submissions/USER_ACTIONS.md."
