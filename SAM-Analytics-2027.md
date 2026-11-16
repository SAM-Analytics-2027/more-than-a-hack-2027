# Technical Report — More than a Hack 2027

* **Team:** `SAM-Analytics-2027` — Matteo Vezzoli, Armando Mio, Stefano Baso
* **Submission file:** `SAM-Analytics-2027.csv` (199,319 rows, SHA-256 `4fdd1f55508dd22209d5997a39c29ed273aa60729496ffb64589ca2023539e21`)
* **Public code repository:** https://github.com/SAM-Analytics-2027/more-than-a-hack-2027
* **Reproduction:** `./reproduce.sh` (Linux/macOS) or `reproduce.bat` (Windows), see §6

---

## 1. Approach

The task is a per-frame labelling problem (`H`, `A`, `D`, `X`) that is ranked on whole possession spells (spellF1). Our pipeline has four stages. Every stage is trained and tuned only on external matches (§3); the competition match is used only as input at prediction time.

1. **Per-frame features from the tactical tracking** (`models/features.py`, 179 features).
   - Only positions are used: ball, players and the team of each track slot. Identities are not needed.
   - The pitch is rotated so that the home team always attacks towards positive x; the direction is found in each half from where the goalkeepers stand.
   - The features describe:
     - the ball: visibility, position, speed, acceleration, distance to the lines, out of the pitch, frames since and until it is seen;
     - the two teams: number of tracked players, centroids, spread, extent, mean speed, share of walking players, defensive lines;
     - the players closest to the ball for each team: distances (also to the last seen ball and to a linear interpolation over short gaps), speed of the closest player, speed towards the ball;
     - touches (ball within 1.8 m of a player): team of the last and next touch, time since and until, touch counts and ball holding over 2 s and 5 s windows, runs of consecutive touches by the same team (passes);
     - centred, past and future window means (0.5 s to 4.5 s) of the main signals.
   - A source flag `tv_source` (1 for the broadcast-derived training matches, 0 otherwise, including the competition match).
2. **Frame-level probabilities** for `H`, `A` and `D`, from two model families.
   - **Trees** (180 columns): scikit-learn `HistGradientBoostingClassifier` and a LightGBM multiclass booster (350 rounds), averaged 50/50.
   - **Temporal convolutional networks** (`models/tcn.py`, 142 columns): three non-causal 1-D dilated convolutional networks on the feature sequence at 5 fps, with an indicator for every missing value. Each has 7 residual blocks with dilations 1–64 and 96 channels, i.e. a receptive field of about ±25 s. The three networks differ only in their random seed; their predictions are interpolated back to 25 fps.
   - **Team symmetry:** every model predicts twice, with the teams as given and with the two teams swapped (pitch rotated by 180°), and the two predictions are averaged. Training uses both versions too.
3. **Combination.** A log-opinion pool: `p ∝ p_trees^0.15 · p_nets^0.85`, renormalised and rounded to 4 decimals. Rounding makes the decoding identical across CPUs and operating systems.
4. **Temporal decoding** (`models/decoder.py`). An explicit-duration Viterbi decoder over `H`, `A` and `D`, run on each half separately, with minimum spell durations, switch costs and the rules of §4. Frames outside the two halves are written as `X`.

**Why this design.**
- spellF1 matches whole spells at IoU thresholds 0.3–0.7. A frame classifier alone flickers: on the external matches the team closest to the ball changes about 30 times per minute, true possessions 3–4 times. The decoder turns frame probabilities into a small number of plausible spells; on the degraded external matches it lifts spellF1 from 0.40 (frame-by-frame argmax) to about 0.68 with the same probabilities.
- The trees carry the local geometric evidence; the networks add tens of seconds of context, which helps most when the ball is missing. In our comparison of model families (same folds and decoder), the networks scored 0.736 against 0.683–0.692 for HGB, LightGBM, XGBoost and a two-stage LightGBM cascade.
- The decoder has a few interpretable parameters, tuned directly on spellF1. macroF1, the tie-break, follows from the same probabilities.

**How missing and noisy tracking is handled.**
- The ball is missing on 41% of the graded frames of the competition tracking, tracks are anonymous and break up. The external tracking is much cleaner, so we **degrade** it to look like the competition tracking before training (§3): missing ball, shifted ball, lost players and re-assigned anonymous slots. The models therefore learn on the same kind of noise they meet at prediction time.
- The features never assume a visible ball: counters since/until the ball was seen, distances to the last seen ball and to its interpolation over short gaps, and touch history carry the information through the gaps. The networks get an explicit indicator for every missing value.
- Frames with no tracked player at all get a neutral cost in the decoder, which bridges them with the surrounding spells.

**Frame alignment.** All computation is on the tactical timeline: row *i* of every feature table is tactical frame *i* (`frame_id − 1` of the ATD). The output has exactly 199,319 rows; the halves come from the competition constants (`games.py`: frames 11,185–90,651 and 110,808–190,197) and everything outside them is `X`. `reproduce.sh` runs `evaluation/validate_submission.py` on the final file.

**Validation protocol.**
- Five folds of whole matches, balanced by source; every external match is predicted by models that never saw it (out-of-fold probabilities).
- The decoder is tuned by coordinate search (2 rounds) on the out-of-fold probabilities of the 10 DFL + Metrica matches.
- **Nested score:** for each fold, the decoder parameters are tuned without that fold and then applied to it.
- The final model is the same recipe trained on all the external matches.

| result | value |
|---|---|
| nested spellF1, 10 DFL + Metrica external matches | **0.7449** |
| mean spellF1 on the same matches with the final decoder parameters (DFL 0.711, Metrica 0.832, worst match 0.647) | 0.747 |
| mean spellF1 on the SkillCorner matches (out-of-fold) | 0.775 |
| previous version (HGB + 3 networks, arithmetic mean, base decoder), nested | 0.7403 |
| public 5-minute window of the competition match (17 possessions; a check of the pipeline, too small to estimate the score) | spellF1 0.7875, macroF1 0.9511 |

The submission file has 315 possessions and the label counts `H` 49,211, `A` 31,954, `D` 77,692, `X` 40,462.

---

## 2. Assets Used and Discarded

**Used**

| asset | how |
|---|---|
| Tactical tracking (`BAR-ATH_tactical_atd.txt` / `.xml`) | the only model input: player and ball positions, team of each slot |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`, Metrica's automatic tagging) | two priors in the decoder: <br>• before each `Kick Off (Start)` instance the ball is not in play yet (`D`); <br>• inside `GOALS` instances, frames with no visible ball and walking players are pushed towards `D`. <br>No other tag is used. |
| Public 5-minute ground truth | only to check the pipeline (frame alignment, file format) |

**Not used**

| asset | reason |
|---|---|
| Tactical and broadcast video | not used: a video model could only be trained and validated on labelled external videos of the same kind, and we had none. |
| Broadcast tracking | a ball fusion is implemented (it fills missing tactical ball positions from the broadcast ball after time alignment, rejecting replays with a player-centroid check), but it is switched off in the submission: no external match has a paired tactical + broadcast feed to validate it on. |
| Homographies | not needed: the tracking is already in pitch coordinates. |

**Variants tested and dropped.** Each variant was scored with the same nested protocol on the 10 DFL + Metrica matches. We required a gain of at least +0.01, consistent across seeds and sources; seed-to-seed noise is about ±0.004.

| variant | result |
|---|---|
| XGBoost, LightGBM alone, two-stage LightGBM cascade ("dead ball?" then "which team?") | 0.686 / 0.683 / 0.687 vs 0.692 for HGB (single families, before ensembling) |
| SkillCorner matches mixed in at full weight, without the source flag | worse on DFL + Metrica (training on those 10 alone was +0.032 better); the `tv_source` flag and weight 1/4 fix it |
| SkillCorner weight 0 or 1/2 for the networks instead of 1/4 | nested −0.004 / −0.005, single networks identical: kept 1/4 |
| Probability calibration (temperature scaling) | all pieces already calibrated (expected calibration error 0.3–1.6%): no change |
| Explicit duration prior on spells (hidden semi-Markov decoder) | −0.0006 |
| Change-hazard decoder (an event classifier feeding the switch costs) | +0.0010 |
| Per-source decoder parameters | +0.0009 / −0.0019 |
| Multinomial stacking of trees and networks with temporal context | −0.0036 |
| Temporal smoothing of the tree probabilities | −0.0016 |
| Test-time averaging over the 5 sub-sampling phases of the networks | −0.0008 |
| Multi-stage refinement of the networks (MS-TCN, 3–4 stages) | single networks +0.005, ensembles ≤ +0.002 |
| Self-attention layer after the convolutional blocks | −0.0049 |
| Dilations 1-3-9-27-81 instead of powers of 2 | −0.0088 |
| Larger ensembles (6–27 networks) | plateau, ±0.002 |
| Networks at 12.5 Hz instead of 5 Hz | +0.002, 3× slower |
| Source flag for DFL-style labels | −0.0002 |
| Stronger weights on short spells | −0.0085 |
| Auxiliary "clean ball" head in the networks | −0.0051 |
| DFL labels with ricochets removed (`models/dfl_ricochet_filter.py`) | not adopted |

---

## 3. Training Data

No frame of the provided game, and no label of any kind about it, was used to train or tune any part of the pipeline. All labels below were produced by the data providers or by our code; none was produced by hand by us.

| dataset | source and licence | matches and amount | labels: who and how |
|---|---|---|---|
| **DFL / IDSSE open data** (Bundesliga and 2. Bundesliga 2022/23) | Bassek et al. (2025), *Scientific Data* 12:195, doi:10.1038/s41597-025-04505-y; CC BY 4.0, credit: Deutsche Fußball Liga | 7 matches, optical tracking at 25 fps; 11.1 h labelled `H`/`A`/`D`, about 3,060 possessions | the ball-status and ball-possession channels of the official positional data, produced by the league's data provider (DFL / Sportec Solutions); translated by our script (`data_prep/prepare_dfl.py`): ball not in play → `D`, otherwise the team in possession |
| **Metrica Sports sample data** (`Sample_Game_1–3`) | github.com/metrica-sports/sample-data | 3 matches, optical tracking at 25 fps; 4.7 h, about 1,030 possessions | built by our code from Metrica's public event data (events annotated by Metrica Sports), with the rules below |
| **SkillCorner open data** (A-League 2024/25) | github.com/SkillCorner/opendata, MIT | 20 matches (19 in the final tree models), broadcast tracking at 10 fps interpolated to 25 fps; 20.7 h, about 5,000 possessions | SkillCorner's *phases of play*, produced by SkillCorner; translated by our script (`data_prep/prepare_skillcorner.py`): inside a phase, the team in possession; outside, `D`; gaps shorter than 2 s between two phases with the ball on the pitch keep the last team; replay and close-up frames become `X` (not used) |

**Metrica labels** (algorithmic, from the events):
- **`D`:** from `BALL OUT`, a shot off target, a goal, or a foul without advantage, until the restart (`SET PIECE`, or goal kicks recorded as `PASS` / `BALL LOST` with subtype `GOAL KICK`);
- **possession at a restart:** the restarting team;
- **possession changes:** on a `RECOVERY` by the other team; failing that, at that team's first pass, carry or shot after a loose ball; a `CHALLENGE` does not change possession; a goalkeeper catch gives possession to the keeper's team;
- **`X`:** offside phases.

**Degradation of the external tracking** (`data_prep/degradation.py`, fixed random seeds, byte-for-byte reproducible). The labels stay those of the clean version; only the tracking changes:
1. **missing ball:** the ball disappears in 80% of the stoppages (around the stop and the restart) and in about 1.5 in-play gaps per minute of about 4 s, more often in the penalty boxes and near the touchlines;
2. **shifted ball:** a smooth random offset (0.8 m standard deviation for DFL, 0.5 m for SkillCorner, whose ball is already estimated from TV images; Metrica has its own values, tuned the same way) plus a lag of 1–2 frames;
3. **lost players and anonymous slots:** players are lost with the rates measured on the competition tracking (higher when close to another player or when the ball is missing), stay invisible for about 2 s and come back as new anonymous tracks in 24 + 22 slots, as in the competition files.

The parameters were chosen by matching **label-free statistics of the provided tactical tracking** (share of missing-ball frames by player speed, gap counts and lengths, ball-to-nearest-player distance, track lengths, number of tracked players). No label and no video of the provided game was involved.

**Training rows.** One frame in 8 for DFL and Metrica, one in 16 for SkillCorner, so that the 20 SkillCorner matches have a total weight of 1/4 (they count as about 5 matches); every match is also used with the two teams swapped; frames of short possessions get a higher weight, `w = sqrt(median duration / spell duration)` clipped to [1/3, 3]. The tree models of the final model use 29 external matches (one SkillCorner match, `1959846`, was not in the final training set); the three frozen networks were trained on all 30.

**Leakage control.** Folds are whole matches, never frames of the same match; the decoder is tuned on out-of-fold probabilities; the reported score is nested. No training or tuning step uses the provided game or its public 5-minute ground truth.

---

## 4. Possession Rules Implemented

The models learn the label conventions of the training data. The decoder then enforces the following rules, with the values tuned on the 10 external matches (`models/frozen/viterbi_params.json`).

**Who has the ball (`H` vs `A`)**
- A possession spell lasts at least `m_pos` = 5 frames (0.2 s).
- A direct `H` ↔ `A` change costs `turn` = 9.5 (in units of −log probability): the models must show sustained evidence for the other team before the decoder switches.
- The features encode the proxies for *controlled* possession given in the rules: who touches the ball, how long a team keeps touching it, and how many consecutive touches (passes) the same team makes.

**Contested balls, deflections and ricochets**
- A direct `H` ↔ `A` change costs an extra `pen_no_touch` = 2.0 when no player of either team is within `touch_th` = 2.5 m of the ball. Possession therefore does not flip while the ball is travelling or loose, only where a player can actually control it; a touch without control or a deflection does not change possession unless the evidence persists.
- A dead-ball interruption that returns to the same team (`H` → `D` → `H`) pays an extra `pen_same` = 1.0, against false stoppages inside a spell.

**Loose balls and tracking gaps**
- While the ball is missing, the last team with sustained evidence keeps the ball unless the models see a clear change; frames without any tracked player are bridged with the surrounding spells.

**Dead ball (`D`) and restarts**
- A dead-ball spell lasts at least `m_dead` = 55 frames (2.2 s).
- Entering `D` costs `stop` = 5.0 and leaving it costs `rest` = 9.0; a constant `bias_d` = −0.7 favours `D`.
- A possession that ends in `D` before `m_pre` = 50 frames (2 s) pays `pen_pre` = 0.5, against phantom possessions just before a throw-in or a foul.
- The restarting team is whatever the models and the decoder find after the stoppage.
- Kick-offs: before each `Kick Off (Start)` instance of the Smart Tagging the frames are forced to `D`; after goals, frames with no visible ball and walking players inside the `GOALS` instance are pushed towards `D`.

**Abstention (`X`)**
- `X` is written only outside the two halves; inside them every frame gets `H`, `A` or `D`.

There are no frame-specific rules, hand-set constants for the provided game or manual corrections.

---

## 5. Limitations

The observations below come from the external matches.

- **Short spells.** Half of the true spells last less than 4 s, and they are the main source of error: in an oracle analysis on a comparable ensemble, copying the true spells shorter than 2 s would add about +0.10 spellF1, perfect dead-ball detection only about +0.02.
- **Touch versus control.** The sources disagree on whether a single touch by the other team is a change of possession (DFL flips in 43% of such episodes, Metrica in 33%, SkillCorner in 16%), and none of them uses exactly the definition of the competition ground truth (corrected tracking). The models learn a compromise: DFL matches score lowest (0.711 against 0.832 for Metrica).
- **Missing ball in contested phases.** The ball goes missing most often in crowded boxes and aerial duels, exactly where possession changes; there the model relies on player movement and errs more.
- **Domain gap.** The degraded external tracking is close to the competition tracking but not identical (in the degraded data players are lost more often than in the competition tracking); SkillCorner is broadcast tracking with the replays removed; no source has a ball height.
- **Frozen networks and one feature.** The three networks were trained before the last change to the pass-streak feature (`last_streak` now resets when the ball leaves the pitch or after 4 s without touches). They are used frozen, so at prediction time they see a slightly different version of that one input; the scores above already include this effect.
- **Noise of the estimate.** spellF1 differences below about 0.04 are noise on a full match (EVALUATION.md); our nested estimate rests on 10 matches.
- **With more time:** labelled external data closer to the competition definition of controlled possession; a dedicated touch/control classifier (it predicts possession flips with AUC 0.90–0.93 on degraded tracking); retraining the networks on the current features; multi-stage refinement of the networks; validating the broadcast ball fusion on paired tactical and broadcast tracking.

---

## 6. Reproduction

1. **Environment** (Python 3.11, exact versions):
   ```bash
   conda env create -f models/environment.yml
   conda activate mtah
   ```
   or `pip install -r requirements.txt` with Python 3.11. Both pin numpy, pandas, pyarrow, scikit-learn, scipy, joblib, numba, llvmlite, PyTorch and LightGBM.
2. **Data.** Unpack the official bundle into `data/` (or set `MTAH_DATA_DIR`), so that these files exist:
   - `data/tactical/BAR-ATH_tactical_atd.txt`
   - `data/tactical/BAR-ATH_tactical_atd.xml`
   - `data/tactical/BAR-ATH_tactical_smart_tagging.xml`

   `data/` is excluded from the repository.
3. **Run** `./reproduce.sh` (Linux/macOS) or `reproduce.bat` (Windows), from the repository root. In under a minute it:
   - checks the library versions and the SHA-256 of the two tracking files (`python models/reproduce.py check --barca-only`);
   - predicts with the frozen weights in `models/frozen/final/` (`python models/reproduce.py predict`), with no training;
   - writes `SAM-Analytics-2027.csv`;
   - runs `evaluation/validate_submission.py` on it.
4. **Determinism.** Fixed seeds, and probabilities rounded to 4 decimals before decoding. The script compares the SHA-256 of the output with the frozen reference and prints `RESULT: IDENTICAL to the reference`. Expected SHA-256: `4fdd1f55508dd22209d5997a39c29ed273aa60729496ffb64589ca2023539e21`, 315 possessions. The same file was obtained on Windows (Python 3.12) and on macOS (Python 3.11), and from a fresh copy of the repository.
5. **Retraining and external scores** (optional, need the external matches prepared with `data_prep/`):
   - `python models/reproduce.py external` recomputes the out-of-fold answers and scores on the 10 DFL + Metrica matches from the frozen fold models;
   - `python models/reproduce.py train --name <name>` retrains the trees and re-tunes the decoder on the external matches (the frozen networks are reused unless `MV_TRAIN_TCN=1`).
   - DFL and SkillCorner are prepared by `data_prep/prepare_dfl.py`, `data_prep/degradation.py` and `data_prep/prepare_skillcorner.py`; the Metrica labels follow the rules of §3.
