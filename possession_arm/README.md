# possession_arm_v1 — two-stage ball-possession model (XGBoost → Viterbi)

A second, independent ball-possession model for More-than-a-Hack 2027 (Phase 1),
built from scratch and trained **only on external labelled matches**.

## Pipeline

    Stage 1   per-frame posterior P(H/A/D | x_t): XGBoost GBDT, native NaN handling,
              tree_method hist + GPU (device cuda), 10 external matches x (clean +
              dirtied tracking; always the clean ground truth).
    Stage 2   per-half duration-constrained Viterbi (the provided exact DP):
              frame costs = -log posterior (+ optional learned class prior),
              context-dependent transition penalties whose EVERY value is learned
              by coordinate search on holdout posteriors of external matches.

Deliverable CSV: frame,label over 199,319 tactical-video rows, X outside the two
published half blocks. Exactly those X frames equal the ungraded remainder of the
private ground truth count (199,319 - 158,857 = 40,462).

## Reproduction, from zero to the submitted file

Requirements: the repository's `requirements.txt` + `xgboost` (GPU), the
competition data bundle under `data/` and the external labelled matches under
`data/altre_partite/` (see `data/analisi/LEGGIMI.md` for who produced them and how).

    # 0. (device check) XGBoost GPU needs a CUDA device with enough VRAM.
    # 1. features + context masks for every external match file
    python possession_arm/cache.py --force
    # 2. nested leave-one-match-out fits (early stopping on the NEXT fold),
    #    held-out posteriors + default-parameter scores
    python possession_arm/eval_external.py
    # 3. final artifact (round count from the LOO p80 best iterations)
    python possession_arm/train.py --mode final
    # 4. decoder parameter search on LOO holdout posteriors (external-only),
    #    writes artifacts/possession_arm_v1/decoder_params.json + search log
    python possession_arm/search_decoder.py --rounds 4
    # 5. competition prediction + public-GT SANITY report (not a tuning signal)
    python possession_arm/predict.py --out evaluation/results/possession_arm_v1.csv
    # 6. official format validation
    python evaluation/validate_submission.py evaluation/results/possession_arm_v1.csv

Everything between steps 1 and 5 lives in `artifacts/possession_arm_v1/`
(models, metadata, decoder parameters with provenance, search history).

## Module map

| module | role |
|---|---|
| `data_external.py` | loaders for the 10 external matches (Barca ATD format), clean labels always from `*_pulite` |
| `half_blocks.py` | half-time split of the stitched external timelines from goal-zone occupancy (tracking only) |
| `features.py` | Stage-1 feature matrix (ball evidence, provided verified-control evidence, visibility, team dynamics, smoothed context) |
| `context.py` | coarse context tiers for the decoder (impact/controlled/occluded/stop/restart), no Smart Tagging |
| `decode.py` | Stage 2: parameters dataclass, frame costs from posteriors + prior, tier transition costs, per-half `decode_possession_sequence` (provided DP) |
| `cache.py` | one npz per external file (X, labels, context masks), version-tagged filenames |
| `train.py` | XGBoost fitting (nested LOO and the all-matches final artifact) |
| `eval_external.py` | LOO scores with the repository metric; writes held-out posteriors |
| `search_decoder.py` | external-only coordinate search of the Stage-2 parameters |
| `predict.py` | competition match: features → posterior → decode → CSV (+ public-GT sanity report) |

## Compliance notes (rules/CLARIFICATIONS.md, rules/FAQ.md)

- Trained/tuned only on external data: DFL 2022/23 (CC-BY, cleaned by the data-team
  analysis docs) + Metrica sample matches, with the *clean* ground truths. The
  competition match never enters any fit, split, threshold or search objective.
- The public 5-minute GT is used ONLY as a reported sanity number at the end of
  `predict.py`; its score is never fed back into any choice.
- Smart Tagging is NOT used at all in v1: the external matches have no Smart
  Tagging, so tag-derived features would put the competition match outside the
  training distribution.
- Feature-set version tag (`v2_*` cache file names) prevents stale-cache mixing;
  the changed version resets every downstream artifact.
