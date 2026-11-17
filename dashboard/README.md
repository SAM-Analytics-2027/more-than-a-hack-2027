# Visual Comparison of Results (`dashboard/`)

A small, light-themed web app, **independent of the model and read-only**, to compare at a glance:
- the **ground truth of the first 5 minutes** (`evaluation/gt/BAR-ATH_possession_gt_public.csv`);
- our models (`evaluation/results/SAM-model.csv` and any other trial);
- the **4 baselines** provided by the organisers (`baseline_raw`, `baseline_smooth25`, `baseline_smooth50`, `baseline_smooth125`).

> It reads the `.csv` files in memory, **never writes to disk**, does not touch the model and does not change any data.

---

## How to start it

- **Double click (Windows):** open `start_dashboard.bat` in this folder.
- **Or from a terminal** (from the repository root):
  ```bash
  python dashboard/app.py
  ```

When a new `.csv` file appears in `evaluation/results/`, refresh the browser page (`F5`).

---

## What it shows (3 sections)

1. **Possession timelines:** coloured bars aligned frame by frame (`H` blue = Barcelona, `A` orange = Athletic Club, `D` grey = dead ball), with a **green / red** strip under each model (same as / different from the ground truth), mouse-wheel zoom and a live count of **possessions and changes** for each row.
2. **First 5 minutes table (direct comparison with the ground truth):** the first, yellow row holds the true numbers (`17 possessions`, `20 changes`, `300.0 s`); each row below shows, for each file, the official scores (`0–100`), how many frames/seconds it gets wrong (split between *teams swapped* and *dead/live*), how many possessions and changes it finds and how far it is from the ground truth.
3. **Whole match table (1st half + 2nd half):** total possessions, mean duration, total changes and `H / A / D` shares over all the 158,857 graded frames.

The public 5 minutes hold only 17 possessions: they are for checking that a file is right, not for choosing between models.

---

## Second page: external matches (`/external`)

The **External matches (DFL + Metrica)** button at the top opens the comparison on the **10 degraded external matches** (7 DFL + 3 Metrica): about 16 hours of play and 4,000 true possessions. This is the page on which to **choose** between models.

**How it judges:** whole matches, **original labels** of the clean versions (`data/external/<source>_clean/<id>/<id>_possession_gt.csv`), the same official scoring code (`evaluation/possession_metrics.py`), with one `X` frame between the two halves as at half-time in Barça-Athletic.

**How to add a model:** one folder per model in `data/external_results/<model_name>/`, with one file per match (`J03WMX.csv`, …, `Sample_Game_3.csv`) in the same format as the Barça-Athletic file (`frame,label`, one row per frame of the match). `python models/reproduce.py external` writes them for the frozen model. Each match must be predicted by a model that **has not seen it in training** (neither the degraded nor the clean version), leaving it out in turn; otherwise the score is inflated. As with `evaluation/results/`, everything stays in `data/` and GitHub does not see it.

**What it shows:**
1. **Summary:** for each model, the mean spellF1 over the 10 matches, per source (DFL, Metrica), the **stability** (worst – best match), the macroF1 and in how many matches it is the best.
2. **Match by match:** the spellF1 of every model on every match (green border = best on that match); click on a column to open it.
3. **Timelines** of the selected match, as on the first page, with the green/red strip over the whole match.
4. **Table of the selected match**, with the same columns as the 5-minute table but over the whole match.
