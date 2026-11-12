# More than a Hack 2027 — `SAM-Analytics-2027`

Official repository of team **`SAM-Analytics-2027`** for the 1st challenge of **More than a Hack 2027** (based on the [official competition repository](https://github.com/metrica-sports/more-than-a-hack-2027/tree/main)).

For full methodological details, see the technical report in [`SAM-Analytics-2027.md`](SAM-Analytics-2027.md).

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Placement & Challenge Rules Compliance](#data-placement--challenge-rules-compliance)
- [Reproducing the Submission (`SAM-Analytics-2027.csv`)](#reproducing-the-submission-sam-analytics-2027csv)
- [External Training Datasets, Licenses & Citations](#external-training-datasets-licenses--citations)
- [Code License](#code-license)

---

## Environment Setup

### Conda (Linux, macOS, Windows)

If `conda` is not installed on your system, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then create and activate the environment:

```bash
conda env create -f modelli_mv/environment.yml
conda activate mtah
```

---

## Data Placement & Challenge Rules Compliance

* **Non-Redistribution:** In strict compliance with the competition rules, `data/` and generated prediction CSVs are excluded from this repository.
* **Target Match Placement:** Place the official competition tracking bundle inside `data/tactical/` (preserving the original directory layout expected by `games.py` and `modelli_mv/dati.py`).

---

## Reproducing the Submission (`SAM-Analytics-2027.csv`)

**Linux / macOS:**
```bash
chmod +x reproduce.sh
./reproduce.sh
```

**Windows:**
```cmd
reproduce.bat
```

---

## External Training Datasets, Licenses & Citations

| Dataset | Competition / Season | Matches Used | Tracking Source | License | Repository |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **DFL / IDSSE Open Data** | German Bundesliga (1st & 2nd Div., 2022/2023) | 7 | Optical TRACAB (25 fps) | CC BY 4.0 | [spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data) |
| **Metrica Sports Sample Data** | Anonymized Elite Matches | 3 | Optical Tracking (25 fps) | Open Sample (Attribution) | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| **SkillCorner Open Data** | Australian A-League (2024/2025) | 10 | Broadcast Tracking (10 fps) | MIT License | [SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

### Data Attribution & License Notices

1. **DFL / IDSSE Open Data (7 Matches — German Bundesliga 2022/2023)**
   * **Citation:** Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer*. Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y
   * **Repository:** https://github.com/spoho-datascience/idsse-data
   * **License & Credit:** Provided with authorization of the **Deutsche Fußball Liga (DFL)** under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. In accordance with license terms, we explicitly credit the **Deutsche Fußball Liga (DFL)** and cite Bassek et al. (2025).

2. **Metrica Sports Sample Data (3 Matches — `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`)**
   * **Citation:** Metrica Sports (2020–2021). *Metrica Sports Sample Tracking and Event Data* [Dataset]. GitHub. https://github.com/metrica-sports/sample-data
   * **License & Credit:** Released publicly by **Metrica Sports** for research and sports analytics development. Source acknowledged per repository guidelines.

3. **SkillCorner Open Data (10 Matches — Australian A-League 2024/2025)**
   * **Citation:** SkillCorner & PySport (2025). *SkillCorner Open Data: Broadcast Tracking and Dynamic Events (Australian A-League 2024/2025)* [Dataset]. GitHub. https://github.com/SkillCorner/opendata
   * **License & Credit:** Released as a joint open-source initiative by **SkillCorner** and **PySport** under the **MIT License**. We credit **SkillCorner** for providing the broadcast tracking and dynamic event data.

## Code License

The source code in this repository is distributed under the license specified in [`LICENSE`](LICENSE). This code license governs only the software in this repository and does not supersede the individual licenses of the external datasets listed above.