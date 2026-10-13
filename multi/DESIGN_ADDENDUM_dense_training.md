# 3-node SCDT — denser training conditions (protocol)

Written 2026-09-29, before the run (originally in `MultiNode_Epileptor/experiments/multi_dense_train/`).

**Status:** adopted on 2026-09-29 by the user as the final 3-node result. The rule-selected statistic was mean + σ. Criterion (b) (cleaner shape) was not met; the replacement was the user's decision, based on boundary accuracy.

## Question
Does denser normal training data improve the locked 3-node SCDT? The locked model learns its static-condition → dynamics map from only 20 records in 5 regimes.

## The only change
- **Training λ grid:**
  - Before: 5 values (−2.130 … −2.070, step 0.015) × 4 reps = 20 records.
  - Now: **13 values (−2.130 … −2.070, step 0.005) × 4 reps = 52 records**.
- Same range, so the extrapolation claim is unchanged. No new λ coincides with a validation λ.
- Training records are regenerated with the same generator and seed rule (`seed + 1000`, event-free rejection).

## Identical to `마무리/multi`
- Validation and calibration records (copied), and the GT boundary (`gt_boundary.csv`, 0.1755).
- The pre-registered statistic rule. With more regimes it may pick a different leading statistic; that is allowed and will be reported.
- Normal-only BO objective and settings: 40 calls, 12 initial, 120-s validation × 4 rollouts, the same starters.
- Reservoir seed 733900; trough-only departure rule; 36 × 30 × 200-s sweep; static-zero control.
- Only one reservoir, as the user decided (no seed-robustness requirement).

## Replacement rule (fixed now)
Reference values are the locked model re-run in this environment with the same trough rule: error −0.0145, E 0.143, R 0.633, 2/64 false departures.

The new model is proposed only if **all** of the following hold:
- **(a) Boundary:** the crossing lies in extrapolation, and |SCDT 50% − GT| ≤ 0.0145 (in q05 units if q05 is again selected; otherwise GT is mapped through the new calibration regression).
- **(b) Cleaner:** full-sweep E < 0.143 and R < 0.633.
- **(c) Normal behaviour:** 0 false departures in its locked validation, and the control has no crossing.

Replacement needs the user's approval.
