# Block 01 Review - Data Layer
## Claude Code Findings

### dataset.py

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Minor | L229 _split_samples | stride windowing off-by-one, last position never sampled |
| 2 | Major | L236 __getitem__ | crash on empty available_frames (IndexError) |
| 3 | Major | L308 __getitem__ | scene_id uses hash() which is non-deterministic per process |
| 4 | Minor | L159 _split | global np.random.seed(42) pollutes RNG state |
| 5 | Minor | L361 _load_static_attrs | silent unknown object types (no warning) |

### data_adapter.py

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 6 | Minor | L10-15 | fragile dynamic import, no error handling |
| 7 | Minor | L59 get_norm_stats | assumes plain dataset, fails if wrapped |
| 8 | Minor | L57-67 get_norm_stats | returns placeholder values when normalize=False |

## My Judgment

- #2 (empty frames crash): FIX - edge case but could crash on bad samples
- #3 (non-deterministic scene_id): FIX - reproducibility issue
- #1, #4, #5, #6, #7, #8: Minor - record as deferred
