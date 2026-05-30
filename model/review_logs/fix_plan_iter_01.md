# Fix Plan - Iteration 1

## Fixed Issues

| # | Source | Issue | Fix |
|---|--------|-------|-----|
| 1 | Block 01 | dataset.py L236: crash on empty frames | Added _empty_sample() guard |
| 2 | Block 01 | dataset.py L308: non-deterministic scene_id | Changed to hashlib.md5 |
| 3 | Block 02 | temporal.py L50: input_proj never called | Added input_proj calls in encode/decode |
| 4 | Block 03 | decoder.py RGBDecoder: valid_mask ignored | Applied valid_mask to mask_prob before compositing |
| 5 | Block 03 | decoder.py L246: sigmoid(0)=0.5 leak | Changed to masked_fill(-10.0) |
| 6 | Block 03 | decoder.py pair_mask: no static-static exclusion | Added static_flag check |
| 7 | Block 04 | train.py: no checkpoint resume | Added --resume flag with model/optimizer/epoch restore |

## Deferred (Minor/Suggestion)

| Issue | Reason |
|-------|--------|
| dataset.py stride off-by-one | Minor, doesn't affect correctness |
| dataset.py global RNG seed | Minor, only affects split reproducibility |
| dataset.py silent unknown types | Minor, current data has known types |
| data_adapter.py fragile import | Minor, works in practice |
| config.py attr_dim=14 vs 13 | Minor, no crash |
| loss.py pos_weight unused | Minor, dead code |
