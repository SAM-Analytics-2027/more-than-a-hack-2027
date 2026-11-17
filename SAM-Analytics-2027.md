# Technical Report — More than a Hack 2027

* **Team:** `SAM-Analytics-2027` — Stefano Baso, Armando Mio, Matteo Vezzoli
* **Submission file:** `SAM-Analytics-2027.csv`
* **Public code repository:** https://github.com/SAM-Analytics-2027/more-than-a-hack-2027

## Summary

We label every frame of Barcelona–Athletic from the automatic tactical tracking alone. An ensemble of gradient-boosted trees (scikit-learn HistGradientBoosting and LightGBM) and three temporal convolutional networks estimates the probability of `H`, `A` and `D` frame by frame; an explicit-duration Viterbi decoder turns these probabilities into possession spells. Everything is trained and tuned on open external matches (DFL/IDSSE, Metrica, SkillCorner: 30 for the networks, 29 for the tree models) whose tracking we **degrade to reproduce the defects of the competition tracking**: missing ball, shifted ball, anonymous and broken player tracks. No label of the competition match is used. Nested cross-validation on the 10 DFL + Metrica matches gives **spellF1 0.745**; on the public 5-minute window the submission scores spellF1 0.7875 and macroF1 0.9511, a check of the pipeline on 17 possessions rather than an estimate of the full-match score. `reproduce.sh` regenerates the submission bit for bit from the frozen weights in under a minute.

---

## 1. Approach

The task is a per-frame labelling problem (`H`, `A`, `D`, `X`) that is ranked on whole possession spells (spellF1). Our pipeline has four stages. Every stage is trained and tuned only on external matches; the competition match is used only as input at prediction time.

1. **Per-frame features from the tactical tracking** (`models/features.py`, 179 features).
   - Only positions are used: ball, players and the team of each track slot. Identities are not needed.
   - The pitch is rotated so that the home team always attacks towards positive x; the direction is found in each half from where the goalkeepers stand.
   - The features describe:
     - The ball: visibility, position, speed, acceleration, distance to the lines, out of the pitch, frames since and until it is seen;
     - The two teams: number of tracked players, centroids, spread, extent, mean speed, share of walking players, defensive lines;
     - The players closest to the ball for each team: distances (also to the last seen ball and to a linear interpolation over short gaps), speed of the closest player, speed towards the ball;
     - Touches (ball within 1.8 m of a player): team of the last and next touch, time since and until, touch counts and ball holding over 2 s and 5 s windows, runs of consecutive touches by the same team (passes);
     - Centred, past and future window means (0.5 s to 4.5 s) of the main signals.
   - A source flag `tv_source` (1 for the broadcast-derived training matches, 0 otherwise, including the competition match).
2. **Frame-level probabilities** for `H`, `A` and `D`, from two model families.
   - **Trees** (180 columns): scikit-learn `HistGradientBoostingClassifier` and a LightGBM multiclass booster (350 rounds), averaged 50/50.
   - **Temporal convolutional networks** (`models/tcn.py`, 142 columns): three non-causal 1-D dilated convolutional networks on the feature sequence at 5 fps, with an indicator for every missing value. Each has 7 residual blocks with dilations 1–64 and 96 channels, i.e. a receptive field of about ±25 s. The three networks differ only in their random seed; their predictions are interpolated back to 25 fps.
   - **Team symmetry:** every model predicts twice, with the teams as given and with the two teams swapped (pitch rotated by 180°), and the two predictions are averaged. Training uses both versions too.
3. **Combination.** A log-opinion pool: `p ∝ p_trees^0.15 · p_nets^0.85`, renormalised and rounded to 4 decimals. Rounding makes the decoding identical across CPUs and operating systems.
4. **Temporal decoding** (`models/decoder.py`). An explicit-duration Viterbi decoder over `H`, `A` and `D`, run on each half separately, with minimum spell durations, switch costs and the possession rules. Frames outside the two halves are written as `X`.

**Why this design.**
- spellF1 matches whole spells at IoU thresholds 0.3–0.7, so flickering is expensive. On the 10 degraded DFL + Metrica matches the team closest to the visible ball changes about 35 times per minute (27–42), while true possessions change about 4 times (3.3–5.3). The decoder turns frame probabilities into a small number of plausible spells; in our first model it lifted spellF1 from 0.40 (frame-by-frame argmax) to 0.68 with the same probabilities.
- The trees carry the local geometric evidence; the networks add tens of seconds of context, which helps most when the ball is missing. In our comparison of model families (same folds and decoder, mean over the 10 matches) the networks scored 0.736 against 0.683–0.692 for HGB, LightGBM, XGBoost and a two-stage LightGBM cascade, and the combination of both families beats either alone.
- The decoder has a few interpretable parameters, tuned directly on spellF1; macroF1, the tie-break, follows from the same probabilities.

**How missing and noisy tracking is handled.**
- The ball is missing on 41% of the graded frames of the competition tracking, and tracks are anonymous and break up. The external tracking is much cleaner, so we **degrade** it before training: the models learn on the same kind of noise they meet at prediction time.
- The features never assume a visible ball: counters since/until the ball was seen, distances to the last seen ball and to its interpolation over short gaps, and the touch history carry the information through the gaps. The networks get an explicit indicator for every missing value.
- Frames with no tracked player at all get a neutral cost in the decoder, which bridges them with the surrounding spells.

**Frame alignment.** All computation is on the tactical timeline: row *i* of every feature table is tactical frame *i* (`frame_id − 1` of the ATD). The output has exactly 199,319 rows; the halves are the competition ones (`games.py`: frames 11,185–90,651 and 110,808–190,197) and everything outside them is `X`. `reproduce.sh` runs `evaluation/validate_submission.py` on the final file.

**Validation protocol and results.** Five folds of whole matches, balanced by source: every external match is predicted by models that never saw it. The decoder is tuned by coordinate search (2 rounds) on these out-of-fold probabilities of the 10 DFL + Metrica matches; the **nested** score tunes it without the fold being judged. The final model is the same recipe trained on all the available external matches: 29 for the tree models, 30 for the three networks.

| Measure | Value | How to read it |
|---|---|---|
| Nested spellF1, 10 DFL + Metrica matches | **0.7449** | Our main estimate: out-of-fold models, decoder tuned without the judged fold |
| Mean spellF1, same matches, final decoder parameters (DFL 0.711, Metrica 0.832, worst match 0.647) | 0.747 | The decoder was tuned on these matches: slightly optimistic |
| Mean spellF1, SkillCorner matches, out-of-fold | 0.775 | Not used to tune the decoder; broadcast tracking, coarser labels |
| Previous version (HGB + 3 networks, arithmetic mean, base decoder), nested | 0.7403 | The same protocol |
| Public 5 minutes of the competition match | spellF1 0.7875, macroF1 0.9511 | 17 possessions: a check of the pipeline, too small to estimate the score |
| Best organisers' baseline on the same 5 minutes (`baseline_smooth125`) | spellF1 0.4242, macroF1 0.5812 | For context; the published full-match floor is spellF1 0.2433 (`baseline_smooth50`) |

The submission file has 315 possessions and the label counts `H` 49,211, `A` 31,954, `D` 77,692, `X` 40,462.

---

## 2. Assets Used and Discarded

**Used**

| Asset | How |
|---|---|
| Tactical tracking (`BAR-ATH_tactical_atd.txt` / `.xml`) | The only model input: player and ball positions, team of each slot |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`, Metrica's automatic tagging) | Two priors in the decoder: <br>• Before each `Kick Off (Start)` instance the ball is not in play yet (`D`); <br>• Inside `GOALS` instances, frames with no visible ball and walking players are pushed towards `D`. <br>No other tag is used. |
| Public 5-minute ground truth | To check the pipeline (frame alignment, file format); not used for training or tuning |
| Organisers' baseline CSVs | For comparison only |

**Not used**

| Asset | Reason |
|---|---|
| Tactical and broadcast video | A video model could only be trained and validated on labelled external videos of the same kind, and we had none |
| Broadcast tracking | Not used: it is on its own timeline, with cuts and replays, and no external match has paired tactical and broadcast tracking on which a fusion could be validated |
| Homographies | Not needed: the tracking is already in pitch coordinates |

**Variants tested and dropped.** Each variant was scored with the same protocol on the 10 DFL + Metrica matches (nested score unless stated). We required a gain of at least +0.01, consistent across seeds and sources; seed-to-seed noise is about ±0.004. None of the variants below reached it.

| Area | Variant → Change in spellF1 |
|---|---|
| Model families | Trees alone (HGB 0.692, XGBoost 0.686, LightGBM 0.683, two-stage "dead ball? / which team?" cascade 0.687) and networks alone (0.736), mean over the 10 matches → we combine trees and networks |
| Training data | SkillCorner mixed in at full weight without a source flag: worse on DFL + Metrica (training on those 10 alone was +0.032 better) → `tv_source` flag and weight 1/4 · SkillCorner weight 0 or 1/2 for the networks: −0.004 / −0.005 (single networks identical) · a flag for DFL-style labels: −0.0002 · stronger weights on short spells: −0.0085 · DFL labels with ricochets removed (`models/dfl_ricochet_filter.py`): not adopted |
| Decoder | Explicit duration prior on spells (hidden semi-Markov): −0.0006 · change-hazard decoder (an event classifier feeding the switch costs): +0.0010 · per-source parameters: +0.0009 / −0.0019 · temporal smoothing of the tree probabilities: −0.0016 |
| Networks | Multi-stage refinement (MS-TCN, 3–4 stages): single networks +0.005, ensembles ≤ +0.002 · self-attention after the convolutional blocks: −0.0049 · dilations 1-3-9-27-81: −0.0088 · 12.5 Hz instead of 5 Hz: +0.002 at 3× the cost · auxiliary "clean ball" head: −0.0051 |
| Combination | Stacking of trees and networks with temporal context: −0.0036 · larger ensembles (6–27 networks): plateau, ±0.002 · test-time averaging over the 5 sub-sampling phases: −0.0008 · probability calibration: not needed, every piece already calibrated (expected calibration error 0.3–1.6%) |

---

## 3. Training Data

No frame of the provided game, and no label of any kind about it, was used to train or tune any part of the pipeline. All labels below were produced by the data providers or by our code; none was produced by hand by us.

| Dataset | Source and licence | Matches and amount | Labels: who and how |
|---|---|---|---|
| **DFL / IDSSE open data** (Bundesliga and 2. Bundesliga 2022/23) | [github.com/spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data), Bassek et al. (2025), *Scientific Data* 12:195, doi:10.1038/s41597-025-04505-y; CC BY 4.0, credit: Deutsche Fußball Liga | 7 matches, optical tracking at 25 fps; 11.1 h labelled `H`/`A`/`D`, about 3,060 possessions | The ball-status and ball-possession channels of the official positional data, produced by the league's data provider (DFL / Sportec Solutions); translated by our script (`data_prep/prepare_dfl.py`): ball not in play → `D`, otherwise the team in possession |
| **Metrica Sports sample data** (`Sample_Game_1–3`) | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), Open Sample (Attribution) | 3 matches, optical tracking at 25 fps; 4.7 h, about 1,030 possessions | Built by our code from Metrica's public event data (events annotated by Metrica Sports), with the rules below |
| **SkillCorner open data** (A-League 2024/25) | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT License | 20 matches (19 in the final tree models), Broadcast tracking at 10 fps interpolated to 25 fps; 20.7 h, about 5,000 possessions | SkillCorner's *phases of play*, produced by SkillCorner; translated by our script (`data_prep/prepare_skillcorner.py`): inside a phase, the team in possession; outside, `D`; gaps shorter than 2 s between two phases with the ball on the pitch keep the last team; replay and close-up frames become `X` (not used) |

These are **provider-derived labels and algorithmic translations, not human ground truth for the competition definition**. Each source draws the line between touch and control in its own way (DFL changes possession at almost every touch, SkillCorner phases are coarser, Metrica follows its events), and none of them follows exactly the competition definition, which is derived from corrected tracking. We treat them as weak labels: the decoder parameters and the model choices are judged only on how well they recover whole spells.

**Metrica labels** (algorithmic, from the events, half by half):
- **Restarts:** every `SET PIECE`, and every `PASS` / `BALL LOST` with subtype `GOAL KICK`; the restarting team has the ball from that frame;
- **`D`:** from `BALL OUT`, from a `SHOT` with subtype `GOAL` or `OUT`, and from a `BALL LOST` with subtype `FORCED` or `HAND BALL` until the next restart; from a `FAULT RECEIVED` until the free kick or penalty, only if no play event happens in between (no advantage); from `END HALF` to the end of the half;
- **Possession changes:** at a restart or a `RECOVERY` by the other team; if the explicit `RECOVERY` is missing, the first `PASS`, `CARRY` or `SHOT` of the other team after a `BALL LOST` or `CHALLENGE` counts as a recovery; `BALL LOST` and `CHALLENGE` alone do not change possession (the ball stays loose until a control event); between events the possession stays with the last control event;
- **`X` (not used):** offside phases (from the pass to the free kick); stretches without events longer than 30 s, unless the players are almost still (mean speed below 1 m/s) and the ball is visible around the next restart, in which case they are `D`; frames before the first restart and after the last event.

**Degradation of the external tracking** (`data_prep/degradation.py`, fixed random seeds, byte-for-byte reproducible). The labels stay those of the clean version; only the tracking changes. The recipes below were tuned on DFL and reused for SkillCorner (except the ball shift); the Metrica matches, whose ball already flickers in and out much more often than in the competition tracking, have their own values tuned the same way.
1. **Missing ball:** the ball disappears in 80% of the stoppages (around the stop and the restart) and in about 1.5 in-play gaps per minute of about 4 s, more often in the penalty boxes and near the touchlines;
2. **Shifted ball:** a smooth random offset (0.8 m standard deviation for DFL, 0.5 m for SkillCorner, whose ball is already estimated from TV images) plus a lag of 1–2 frames;
3. **Lost players and anonymous slots:** players are lost at 1.4 times the rates measured on the competition tracking (higher when close to another player or when the ball is missing), stay invisible for about 2 s and come back as new anonymous tracks in 24 + 22 slots, as in the competition files.

The parameters were chosen by matching **label-free statistics of the provided tactical tracking** (share of missing-ball frames by player speed, gap counts and lengths, ball-to-nearest-player distance, track lengths, number of tracked players). No label and no video of the provided game was involved.

**Training rows.** One frame in 8 for DFL and Metrica, one in 16 for SkillCorner, so that the 20 SkillCorner matches have a total weight of 1/4 (they count as about 5 matches); every match is also used with the two teams swapped; frames of short possessions get a higher weight, `w = sqrt(median duration / spell duration)` clipped to [1/3, 3]. The tree models of the final model use 29 external matches (one SkillCorner match, `1959846`, was not in the final training set); the three frozen networks were trained on all 30.

**Leakage control.** Folds are whole matches, never frames of the same match; the decoder is tuned on out-of-fold probabilities; the reported score is nested. No training or tuning step uses the provided game or its public 5-minute ground truth.

---

## 4. Possession Rules Implemented

The models learn the label conventions of the training data; the decoder then enforces the rules below, with the values tuned on the 10 external matches (`models/frozen/viterbi_params.json`). Costs are in units of −log probability.

| Situation | How it is treated |
|---|---|
| **Controlled possession** | The features encode the proxies for control given in the rules: who touches the ball (within 1.8 m), how long a team keeps touching it, and how many consecutive touches (passes) the same team makes. A possession spell lasts at least `m_pos` = 5 frames (0.2 s); a direct `H` ↔ `A` change costs `turn` = 9.5, so the models must show sustained evidence for the other team before the decoder switches. |
| **Passes and loose balls** | There is no separate "contested" state: possession stays with the last team through a pass or a loose ball unless the evidence for the other team outweighs the switch cost. While the ball is missing, the features carry the last seen and interpolated ball; frames without any tracked player are bridged with the surrounding spells. |
| **Challenges, deflections and ricochets** | A direct `H` ↔ `A` change costs an extra `pen_no_touch` = 2.0 when no player of either team is within `touch_th` = 2.5 m of the ball: possession does not flip while the ball is travelling or loose, only where a player can actually control it. A touch without control or a deflection does not change possession unless the evidence persists. |
| **Stoppages (`D`)** | A dead-ball spell lasts at least `m_dead` = 55 frames (2.2 s). Entering `D` costs `stop` = 5.0 and leaving it costs `rest` = 9.0; a constant `bias_d` = −0.7 favours `D`. A dead-ball interruption that returns to the same team (`H` → `D` → `H`) pays an extra `pen_same` = 1.0, against false stoppages inside a spell. A possession that ends in `D` before `m_pre` = 50 frames (2 s) pays `pen_pre` = 0.5, against phantom possessions just before a throw-in or a foul. |
| **Restarts** | The restarting team is whatever the models and the decoder find after the stoppage; there is no separate restart-type detector. Kick-offs: before each `Kick Off (Start)` instance of the automatic Smart Tagging the frames are forced to `D`; after goals, frames inside the `GOALS` instance with no visible ball and walking players are pushed towards `D`. |
| **Halves and abstention (`X`)** | Features and decoding never cross half-time. `X` is written only outside the two halves; inside them every frame gets `H`, `A` or `D`. |

There are no frame-specific rules, hand-set constants for the provided game or manual corrections.

---

## 5. Limitations

The observations below come from the external matches.

- **Short spells.** Half of the true spells last less than 4 s, and they are the main source of error: in an oracle analysis on a comparable ensemble, copying the true spells shorter than 2 s would add about +0.10 spellF1, perfect dead-ball detection only about +0.02.
- **Touch versus control.** The sources disagree on whether a single touch by the other team is a change of possession (DFL flips in 43% of such episodes, Metrica in 33%, SkillCorner in 16%), and none uses exactly the definition of the competition ground truth. The models learn a compromise: DFL matches score lowest (0.711 against 0.832 for Metrica).
- **Missing ball in contested phases.** The ball goes missing most often in crowded boxes and aerial duels, exactly where possession changes; there the model relies on player movement and errs more.
- **Domain gap.** The degraded external tracking is close to the competition tracking but not identical (in the degraded data players are lost more often than in the competition tracking); SkillCorner is broadcast tracking with the replays removed; no source has a ball height.
- **Noise of the estimate.** spellF1 differences below about 0.04 are noise on a full match (EVALUATION.md); our nested estimate rests on 10 matches.
- **With more time:** training labels closer to the competition definition of controlled possession, which would address the main disagreement between our sources; retraining the networks together with the trees; filling the missing ball from the broadcast tracking, once it can be validated on matches with both feeds.

---

## 6. Reproduction

1. **Environment** (Python 3.11, exact versions of numpy, pandas, pyarrow, scikit-learn, scipy, joblib, numba, llvmlite, PyTorch and LightGBM):
   ```bash
   conda env create -f models/environment.yml
   conda activate mtah
   ```
   or `pip install -r requirements.txt` with Python 3.11.
2. **Data.** Unpack the official bundle into `data/` (or set `MTAH_DATA_DIR`), so that `data/tactical/BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml` exist. `data/` is excluded from the repository.
3. **Run** `./reproduce.sh` (Linux/macOS) or `reproduce.bat` (Windows) from the repository root. In under a minute it checks the library versions and the SHA-256 of the two tracking files (`python models/reproduce.py check --barca-only`), predicts with the frozen weights in `models/frozen/final/` with no training (`python models/reproduce.py predict`), writes `SAM-Analytics-2027.csv` and validates it with `evaluation/validate_submission.py`. Step 1 stops the script if Python or a library version differs from the pinned ones, so that the file is produced only in the environment in which it is guaranteed identical.
4. **Determinism.** Fixed seeds, and probabilities rounded to 4 decimals before decoding. The script compares the SHA-256 of the output with the frozen reference and prints `RESULT: IDENTICAL to the reference`; expected SHA-256 `4fdd1f55508dd22209d5997a39c29ed273aa60729496ffb64589ca2023539e21`, 315 possessions. The same file was obtained on Windows and on macOS, and from a fresh copy of the repository.
