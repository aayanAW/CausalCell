#!/usr/bin/env bash
# Phase 11: reproducibility verification on a fresh worktree.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

WORKTREE="/tmp/perturbcausal-repro-$(date +%Y%m%d%H%M)"
echo "  Creating fresh worktree at $WORKTREE"
git worktree add "$WORKTREE" v4-execution 2>&1 | tail -5

cd "$WORKTREE"

# Try the Makefile if it exists
if [ -f "Makefile" ]; then
    timeout 1800 make env 2>&1 | tail -10 || echo "  make env failed"
fi

# Compare a key JSON
src_state="$ROOT/results/state/state_f1_summary.json"
worktree_state="$WORKTREE/results/state/state_f1_summary.json"
if [ -f "$src_state" ] && [ -f "$worktree_state" ]; then
    python3 -c "
import json
with open('$src_state') as f: a = json.load(f)
with open('$worktree_state') as f: b = json.load(f)
f1_a = a.get('state',{}).get('f1')
f1_b = b.get('state',{}).get('f1')
print(f'  state F1: src={f1_a} worktree={f1_b} diff={abs(f1_a-f1_b) if f1_a and f1_b else \"N/A\"}')
"
fi

cd "$ROOT"
echo "  Worktree available at $WORKTREE for manual inspection."
echo "  Cleanup with: git worktree remove $WORKTREE"
