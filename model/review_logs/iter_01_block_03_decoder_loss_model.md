# Block 03 Review - Decoder/Loss/Model
## Claude Code Findings

### decoder.py + loss.py

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Major | decoder.py RGBDecoder.forward | valid_mask ignored in RGB synthesis, padding objects contribute |
| 2 | Major | decoder.py MultiHeadDecoder.forward | pair_mask doesn't exclude static-static pairs |
| 3 | Major | decoder.py L246 | mask_logits zeroed → sigmoid(0)=0.5 leaks into RGB, should use -10 |
| 4 | Minor | loss.py L139 | pos_weight computed but never passed to BCE |

## My Judgment
- #1 (valid_mask in RGB): FIX - padding objects should not contribute
- #2 (static-static pair): FIX - static objects never collide
- #3 (sigmoid leak): FIX - use large negative logits
- #4 (pos_weight): defer - dead code, no crash
