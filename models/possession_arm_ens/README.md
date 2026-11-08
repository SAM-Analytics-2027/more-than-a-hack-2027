# possession_arm_ens — posterior-averaging ensemble + shared Viterbi

Model family 7 — the closer. It averages the held-out posteriors of the
ingredient families per nested-LOO fold (each ingredient already ran the
identical protocol), decodes the AVERAGE with the shared Viterbi, and searches
its own decoder parameters externally on those composed posteriors. No new fit
happens anywhere in this family; competition-time posteriors are the
equal-weight mean of the chosen ingredients' FINAL artifacts.

## Chosen composition: lgbm + catboost + crf (equal weights)

Ingredient-choice screens ran under the same external-only rule as every other
tuning decision in this repo (default-decode screen + 1 search round each,
external LOO mean spellF1 over the 20 held-out files):

| composition | default | post-search (rounds) |
|---|---|---|
| all six | 0.6029 | 0.6314 (4) |
| catboost+crf | 0.6345 | 0.6528 (1) |
| v1+catboost+crf | 0.6316 | 0.6537 (1) |
| **lgbm+catboost+crf** | **0.6345** | **0.6539 (1)** |
| lgbm+catboost+crf+v1 | 0.6283 | — |
| all six + lgbm (4+tcn) | 0.6136 | — |

(TCN 0.5104 and seg 0.4755 dilute the average; v1 0.6298 adds little over the
two GBDTs already in. The trio keeps the leading CRF 0.6448 and both GBDT
posteriors for diversity.) Full `--rounds 4` rerun of the trio composition is
the number recorded in `artifacts/decoder_params.json` and the scoreboard.

| step | script | notes |
|---|---|---|
| 1 | (none) | relies on every ingredient's LOO posteriors in `data/_arm_cache/` |
| 2 | `python models/possession_arm_ens/train.py --mode loo` | composes + scores per fold; writes `looP_v2_possession_arm_ens_*` |
| 3 | `python models/possession_arm_ens/search_decoder.py --rounds 4` | shared coordinate descent on the composed posteriors |
| 4 | `python models/possession_arm_ens/predict.py --out evaluation/results/possession_arm_ens.csv` | submission + public-GT sanity |
| 5 | `python evaluation/validate_submission.py evaluation/results/possession_arm_ens.csv` | format check |

`--families` selects the blend (default: lgbm+catboost+crf, the validated
composition; use `--families possession_arm_v1 ...` to reproduce the screens).
Missing ingredient posteriors or final artifacts fail fast with an explicit
list. Selection of the blend is done ONLY on external LOO spellF1 (the
ingredients' own numbers), never on the public sanity nor the competition
match.

## Documented negative: the rule ingredient + log-space blend

`--weights W1 ... Wn` and `--log-blend` (geometric rows ∝ ∏ P_f ** w_f; the
log blend preserves each ingredient's log-dominance instead of diluting it)
were added for the possession_arm_rule ingredient experiment:

| blend | default | post-search (4) | artifact |
|---|---|---|---|
| trio (equal arithmetic) | 0.6324 | **0.6543** | `artifacts/decoder_params.json` |
| trio+rule γ=0.05 log-blend | 0.6367 | 0.6537 | `artifacts/logblend_g005/` (kept as negative) |
| trio+rule γ ∈ {0.025…0.3} screens | 0.6243–0.6367 (peak 0.05) | — | — |
| trio+rule prob-space mean/weights | 0.5416–0.6158 | — | — |

Probability averaging dilutes the rule's peaky D mass; the geometric blend
recovers it but the searched optimum stays 0.0006 below the trio's — rule
in blend not selected, consistent with the rule family's own external LOO
(0.4770). Production composition and posteriors remain the validated trio.
Screens overwrite the ens tag + `loo_rounds_summary.json`, so the trio LOO
must be re-run LAST (restore step) after any screen.
