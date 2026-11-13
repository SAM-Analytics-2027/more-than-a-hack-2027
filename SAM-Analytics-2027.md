# Technical Report — More than a Hack 2027

* **Team Name:** `SAM-Analytics-2027`
* **Submission File:** `SAM-Analytics-2027.csv`
* **Public Code Repository:** `https://github.com/SAM-Analytics-2027/more-than-a-hack-2027`

---

## 1. Approach

* **What to include:**
  * High-level overview of your modeling architecture and pipeline stages (e.g., feature extraction, frame-level probabilistic modeling, and temporal sequence decoding).
  * Technical rationale explaining why this design is suited for football possession segmentation.
  * How your pipeline enforces exact frame-level alignment with the `199,319`-frame tactical timeline (`25 fps`).
  * How your method addresses the competition evaluation metrics: `spellF1` (primary ranking metric based on temporal IoU) and `macroF1` (tie-breaker).
* **What NOT to include:**
  * Do **not** state or imply that architectural choices or hyperparameters were tuned on `BAR-ATH` or on the 5-minute public ground truth (`BAR-ATH_possession_gt_public.csv`).
  * Do **not** include raw frame predictions or confidential match statistics from `BAR-ATH`.

> **Write here:** You must describe your overall system architecture, explain why you chose this approach, detail how you guarantee exact alignment to the 199,319 tactical frames, and explain how your design optimizes `spellF1` and `macroF1`.

---

## 2. Assets Used and Discarded

* **What to include:**
  * Complete list of official competition assets used (e.g., tactical optical tracking data, pitch dimensions) and the categories of features extracted from them.
  * How multiple assets or feature representations were integrated into the pipeline.
  * Every asset, data stream, or model family that was tested and subsequently discarded (e.g., broadcast video, smart tagging, alternative classifiers), along with clear technical reasons for discarding each one.
* **What NOT to include:**
  * Do **not** mention any manual annotations on `BAR-ATH`, external event feeds for `BAR-ATH` (Opta, StatsBomb, Wyscout, LaLiga, etc.), or any insights derived from watching the `BAR-ATH` video.

> **Write here:** You must list all official and derived assets you used (and how they were integrated), followed by all assets, video streams, or candidate models you tested and discarded together with the reasons why.

---

## 3. Training Data

* **What to include:**
  * Complete inventory of all external labeled datasets used for training, cross-validation, and hyperparameter tuning.
  * For each dataset: official name, repository/DOI source, who created the annotations, exact number of matches/frames used, sampling weights, and any label mapping or filtering applied.
  * Explicit confirmation of cross-validation strategy and strict data-leakage prevention (`BAR-ATH` and `BAR-ATH_possession_gt_public.csv` never used for training, validation, or parameter tuning).
* **What NOT to include:**
  * Never include `BAR-ATH` data (full match or public 5-minute slice) in training, validation, or threshold selection.
  * Do **not** paste raw tracking rows, event logs, or downloadable links to redistributed match data.

> **Write here:** You must list every external training dataset (source, citation/link, annotator origin, and exact match/frame quantities), explain how labels were mapped/filtered, and confirm that zero `BAR-ATH` data was used for training or tuning.

---

## 4. Possession Rules Implemented

* **What to include:**
  * How active possession (`H` vs. `A`) is determined and sustained across frames.
  * How contested balls, 50/50 duels, tackles, and deflections/ricochets are handled without causing false possession flips.
  * How loose balls and temporary ball-tracking dropouts are bridged.
  * How dead-ball stoppages (`D`) and play restarts (throw-ins, corners, free kicks, kick-offs) are detected and constrained.
  * When and how abstention (`X`) is emitted (e.g., outside regulation playing halves).
* **What NOT to include:**
  * Do **not** include match-specific hardcoded frame rules, manual overrides, or post-hoc corrections tailored to `BAR-ATH`.

> **Write here:** You must explain the exact modeling and decoding rules governing `H` vs. `A` possession, contested balls and deflections, loose balls/tracking gaps, dead-ball states (`D`) and restarts, and abstentions (`X`).

---

## 5. Limitations

* **What to include:**
  * Candid analysis of known failure modes and edge cases (e.g., domain gap between broadcast training data and tactical inference, latency on quick restarts, crowded penalty-box scrambles, lack of 3D ball height).
  * Concrete, actionable improvements you would pursue given more time or additional data.
* **What NOT to include:**
  * Do **not** include error analyses, frame timestamps, or observations derived from visually comparing your predictions against the `BAR-ATH` video (strictly prohibited by `CLARIFICATIONS.md`).

> **Write here:** You must describe where your system fails or loses accuracy (edge cases, tracking noise, transition delays, domain shift) and what specific improvements you would implement with more time.

---

## 6. Reproducibility

* **What to include:**
  * Exact Python version and pinned dependency installation steps (`modelli_mv/environment.yml` and `requirements.txt`).
  * Instructions on where to place the official `data/` bundle (noting it is excluded from the repository).
  * Determinism guarantees (fixed random seeds, probability rounding, frozen weights/parameters).
  * Exact commands or wrapper scripts (`./reproduce.sh` on Linux/macOS, `reproduce.bat` on Windows) to regenerate `SAM-Analytics-2027.csv` byte-for-byte and verify it with `evaluation/validate_submission.py`.
* **What NOT to include:**
  * Do **not** include links to unauthorized copies of the competition `data/` bundle or steps requiring manual intervention.

> **Write here:** You must provide exact step-by-step instructions to recreate the pinned environment, place the official data folder, run the deterministic reproduction script (`reproduce.sh` / `reproduce.bat`), and validate `SAM-Analytics-2027.csv`.