# More than a Hack 2027 — `SAM-Analytics-2027`

Repository of team **`SAM-Analytics-2027`** for the first challenge of **More than a Hack 2027** (built on the [official competition repository](https://github.com/metrica-sports/more-than-a-hack-2027/tree/main)).

The method is described in the technical report [`SAM-Analytics-2027.md`](SAM-Analytics-2027.md).

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Placement](#data-placement)
- [Reproducing the Submission (`SAM-Analytics-2027.csv`)](#reproducing-the-submission-sam-analytics-2027csv)
- [Other Commands](#other-commands)
- [Repository Layout](#repository-layout)
- [External Training Datasets, Licenses & Citations](#external-training-datasets-licenses--citations)
- [Code License](#code-license)

---

## Environment Setup

### Conda (Linux, macOS, Windows)

If `conda` is not installed on your system, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then create and activate the environment:

```bash
conda env create -f models/environment.yml
conda activate mtah
```

### pip

With Python 3.11:

```bash
pip install -r requirements.txt
```

Both files pin the exact versions (Python 3.11, numpy, pandas, pyarrow, scikit-learn, scipy, joblib, numba, llvmlite, PyTorch, LightGBM). With these versions the submission file is reproduced bit for bit; the same file was also obtained on Windows with Python 3.12.

---

## Data Placement

* **Competition match (needed to reproduce the submission):** unpack the official competition bundle into `data/`, keeping its names, so that `data/tactical/BAR-ATH_tactical_atd.txt`, `data/tactical/BAR-ATH_tactical_atd.xml` and `data/tactical/BAR-ATH_tactical_smart_tagging.xml` exist. `MTAH_DATA_DIR` can point to another folder.
* **External matches (needed only to retrain or to score on the external matches):** 30 open matches, prepared with the scripts in [`data_prep/`](data_prep/) into `data/external/<source>_clean/<id>/` (tracking + labels) and `data/external/<source>_degraded/<id>/` (the same tracking with the defects of the competition tracking). See [External Training Datasets](#external-training-datasets-licenses--citations) and the technical report.
* **Non-redistribution:** in compliance with the competition rules, `data/` and every generated prediction CSV (including `SAM-Analytics-2027.csv`) are excluded from this repository by `.gitignore`.

---

## Reproducing the Submission (`SAM-Analytics-2027.csv`)

With the `mtah` environment active, run the reproduction script from the repository root:

**Linux / macOS:**
```bash
chmod +x reproduce.sh
./reproduce.sh
```

**Windows:**
```bat
.\reproduce.bat
```

> **Note**: in Windows PowerShell or VS Code (instead of a Conda terminal), make sure `conda init` has been run, then activate the environment (`conda activate mtah`) and run `.\reproduce.bat`.

The script runs four steps in under a minute:

1. `python models/reproduce.py check --barca-only` checks the library versions and the SHA-256 fingerprints of the two competition tracking files;
2. `python models/reproduce.py predict` computes the features of the competition match, applies the frozen model in `models/frozen/` and writes `evaluation/results/SAM-model.csv`; it prints `RESULT: IDENTICAL to the reference` when the file has the expected SHA-256 fingerprint;
3. copies it to `SAM-Analytics-2027.csv`;
4. validates it with `evaluation/validate_submission.py`.

---

## Other Commands

All from the repository root, with the `mtah` environment active:

| command | what it does |
|---|---|
| `python models/reproduce.py check` | library versions and fingerprints of all 92 data files (competition match and external matches) |
| `python models/reproduce.py predict` | Barça–Athletic CSV from the frozen weights (`evaluation/results/SAM-model.csv`) |
| `python models/reproduce.py external` | out-of-fold answers on the 10 DFL + Metrica matches from the frozen fold models (`data/external_results/SAM-model/`), with their spellF1 |
| `python models/reproduce.py train --name NAME` | retrains everything from the external matches: 5 folds, Viterbi tuning, final model (weights in `data/cache/models/NAME/`) |
| `python models/reproduce.py predict --model NAME` | the same as `predict`, with the weights of trial `NAME` |
| `python dashboard/app.py` | local read-only dashboard comparing the CSVs (see [`dashboard/README.md`](dashboard/README.md)) |
| `python data_prep/prepare_dfl.py` | downloads and prepares the 7 DFL matches (clean version) |
| `python data_prep/degradation.py` | degrades the 7 clean DFL matches |
| `python data_prep/prepare_skillcorner.py` | downloads and prepares the 20 SkillCorner matches (clean and degraded) |

The scripts in `data_prep/` need `kloppy` and the versions in `data_prep/requirements.txt`. The 3 Metrica matches are prepared with the rules described in the technical report.

The first run of `external` or `train` builds a feature cache in `data/cache/` (a few minutes); the cache is rebuilt automatically whenever the feature code changes.

---

## Repository Layout

```
models/                     the possession model
  reproduce.py              entry point: check, predict, external, train
  features.py               per-frame features from the tracking
  epts_reader.py            reader of the competition tracking format (FIFA EPTS)
  dataset.py                the 30 external matches and the feature cache
  cross_validation.py       folds, training rows, HGB, Viterbi tuning
  model_families.py         LightGBM / XGBoost families and training rows
  tcn.py                    temporal convolutional network
  decoder.py                Viterbi decoder with minimum durations, and the score
  dfl_ricochet_filter.py    optional ricochet filter for the DFL labels (experiment)
  constants.py              every constant and hyperparameter
  environment.yml           conda environment
  frozen/                   the frozen model: final/, folds/fold0..4/, Viterbi parameters, expected fingerprints
data_prep/                  scripts that prepare and degrade the external matches
dashboard/                  read-only comparison web app
evaluation/, starter/, rules/, games.py   official competition files
reproduce.sh, reproduce.bat              one-command reproduction
SAM-Analytics-2027.md                    technical report
```

---

## External Training Datasets, Licenses & Citations

| Dataset | Competition / Season | Matches Used | Tracking Source | License | Repository |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **DFL / IDSSE Open Data** | German Bundesliga (1st & 2nd Div., 2022/2023) | 7 | Optical TRACAB (25 fps) | CC BY 4.0 | [spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data) |
| **Metrica Sports Sample Data** | Anonymized Elite Matches | 3 | Optical Tracking (25 fps) | Open Sample (Attribution) | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| **SkillCorner Open Data** | Australian A-League (2024/2025) | 20 (19 in the final model) | Broadcast Tracking (10 fps) | MIT License | [SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

### Data Attribution & License Notices

1. **DFL / IDSSE Open Data (7 Matches — German Bundesliga 2022/2023)**
   * **Citation:** Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer*. Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y
   * **Repository:** https://github.com/spoho-datascience/idsse-data
   * **License & Credit:** Provided with authorization of the **Deutsche Fußball Liga (DFL)** under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. In accordance with license terms, we explicitly credit the **Deutsche Fußball Liga (DFL)** and cite Bassek et al. (2025).

2. **Metrica Sports Sample Data (3 Matches — `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`)**
   * **Citation:** Metrica Sports (2020–2021). *Metrica Sports Sample Tracking and Event Data* [Dataset]. GitHub. https://github.com/metrica-sports/sample-data
   * **License & Credit:** Released publicly by **Metrica Sports** for research and sports analytics development. Source acknowledged per repository guidelines.

3. **SkillCorner Open Data (20 Matches — Australian A-League 2024/2025)**
   * **Citation:** SkillCorner & PySport (2025). *SkillCorner Open Data: Broadcast Tracking and Dynamic Events (Australian A-League 2024/2025)* [Dataset]. GitHub. https://github.com/SkillCorner/opendata
   * **License & Credit:** Released as a joint open-source initiative by **SkillCorner** and **PySport** under the **MIT License**. We credit **SkillCorner** for providing the broadcast tracking and dynamic event data.

## Code License

The source code in this repository is distributed under the license specified in [`LICENSE`](LICENSE). This code license governs only the software in this repository and does not supersede the individual licenses of the external datasets listed above.
