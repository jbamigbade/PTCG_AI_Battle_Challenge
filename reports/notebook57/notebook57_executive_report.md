# Notebook 57 — Probability Attribution and Feature Importance

## Executive Findings

- **Predicted action stability**: All four frozen policies preserved identical predicted actions across all 184 controlled evaluation cases.
- **Probability sensitivity**: The largest observed case-level probability drift was 0.168.
- **Most important shared feature**: numeric__is_legal__pass had the highest mean importance when retained across the strategy set (0.0554).
- **Dominant R3 feature**: categorical__active_card_Bulbasaur was the strongest feature in R3_STATE_CENTRIC_POLICY with importance 0.0863.
- **Largest ablation stage**: R2_TO_R3 removed 30.22% of the prior model importance.
- **Most sensitive action on average**: Quick Attack had the largest mean probability drift (0.0534).
- **Largest single-action drift**: Ascension produced the largest observed single-action probability drift (0.168).
- **Primary probability gain**: Ascension gained the most probability on average (+0.0471).
- **Primary probability loss**: Quick Attack lost the most probability on average (-0.0475).
- **Most drift-sensitive variant**: V03 had the largest mean case-level drift (0.0776).
- **Probability mass conservation**: Signed action shifts sum to approximately zero for every strategy pair, confirming numerically stable probability redistribution.
- **Interpretability conclusion**: Removing explicit legality and compatibility signals changes policy confidence substantially while preserving the decision boundary.

## Policy Recommendation

- **Primary candidate — R3_STATE_CENTRIC_POLICY**: R3 preserves action decisions across the controlled benchmark while relying more strongly on intrinsic battle-state features rather than handcrafted legality/signature signals.
- **Validation requirement — R3_STATE_CENTRIC_POLICY**: R3 should proceed to robustness, calibration, SHAP, and large-scale tournament evaluation before final submission selection.
- **Fallback reference — R0_CURRENT_BASELINE**: R0 remains the reference policy for measuring the effect of progressive feature removal and confidence redistribution.

## Notebook Status

**PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE**
