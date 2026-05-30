# Block 04 Review - Train/GPU/Checkpoint
## Claude Code Findings

| # | Severity | Issue |
|---|----------|-------|
| 1 | Major | No checkpoint resume (--resume flag missing) |
| 2 | Minor | test loader not explicitly deleted before OOM retry |

## My Judgment
- #1 (no resume): Major but defer - current goal is smoke train proof, not full training
- #2 (cleanup): Minor, defer

OOM auto-reduction, AMP, checkpoint save, real data usage, device handling all verified correct.
