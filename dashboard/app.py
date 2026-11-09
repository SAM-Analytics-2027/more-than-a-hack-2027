"""Read-only Visual Comparison WebApp for More than a Hack 2027.

Completely independent from any possession algorithm:
  - Never modifies data/, evaluation/, or algorithm files (100% read-only in RAM).
  - Automatically discovers every *.csv in evaluation/results/ and compares them against
    evaluation/gt/BAR-ATH_possession_gt_public.csv (exact 5-minute solution) and across
    the entire 90-minute match.
  - Second page (/esterne): the 10 external matches (7 DFL + 3 Metrica, dirtied like BAR-ATH),
    comparing every model folder in data/risultati_esterne/ against the original labels over
    whole matches. This is the page to CHOOSE models on; the 5 minutes are for checking only.

Usage:
    python dashboard/app.py
    python dashboard/app.py --port 8765 --no-browser
"""

import argparse
import glob
import json
import os
import sys
import threading
import webbrowser
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL_DIR = os.path.join(ROOT, "evaluation")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402


def _rle_encode(codes: np.ndarray) -> list:
    """Encode 1D int8 label array into compact [[start, end_exclusive, code], ...] list."""
    if len(codes) == 0:
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    return [
        [int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])]
        for i in range(len(bounds) - 1)
    ]


def _full_match_stats(codes: np.ndarray, graded_runs: list, fps: float) -> dict:
    """Compute clear, human-readable counts across both halves (158,857 graded frames)."""
    filled = pm.fill_abstentions(codes, graded_runs)
    spells = pm.possession_spells(filled, graded_runs)
    spells_h1 = pm.possession_spells(filled, [graded_runs[0]])
    spells_h2 = pm.possession_spells(filled, [graded_runs[1]])
    trans_all = pm.transitions(filled, graded_runs)

    durations_s = np.array([(b - a) / fps for a, b, _ in spells], dtype=np.float64)
    n_spells_h = sum(1 for _, _, k in spells if k == pm.H)
    n_spells_a = sum(1 for _, _, k in spells if k == pm.A)

    graded_mask = np.zeros(len(codes), dtype=bool)
    for s, e in graded_runs:
        graded_mask[s:e] = True
    g_codes = codes[graded_mask]
    n_graded = max(1, len(g_codes))

    cnt_h = int(np.sum(g_codes == pm.H))
    cnt_a = int(np.sum(g_codes == pm.A))
    cnt_d = int(np.sum(g_codes == pm.D))
    cnt_x = int(np.sum(g_codes == pm.X))
    alive = cnt_h + cnt_a

    return {
        "total_graded_s": n_graded / fps,
        "n_spells_full": len(spells),
        "n_spells_h1": len(spells_h1),
        "n_spells_h2": len(spells_h2),
        "n_spells_H": n_spells_h,
        "n_spells_A": n_spells_a,
        "n_trans_full": len(trans_all),
        "mean_dur_full_s": float(np.mean(durations_s)) if len(durations_s) else 0.0,
        "median_dur_full_s": float(np.median(durations_s)) if len(durations_s) else 0.0,
        "sec_H": cnt_h / fps,
        "sec_A": cnt_a / fps,
        "sec_D": cnt_d / fps,
        "sec_X": cnt_x / fps,
        "pct_H_graded": 100.0 * cnt_h / n_graded,
        "pct_A_graded": 100.0 * cnt_a / n_graded,
        "pct_D_graded": 100.0 * cnt_d / n_graded,
        "pct_X_graded": 100.0 * cnt_x / n_graded,
        "home_share_full": 100.0 * cnt_h / alive if alive > 0 else 0.0,
    }


def build_dashboard_payload() -> dict:
    """Read GT and all submissions in evaluation/results/ into a single JSON-serializable dict."""
    gt_path = os.path.join(EVAL_DIR, "gt", "BAR-ATH_possession_gt_public.csv")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Public ground truth not found: {gt_path}")

    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    n_frames = len(gt_codes)
    fps = float(GAMES[GAME]["fps"])
    half_blocks = half_video_blocks(GAME)  # [(1, 11185, 90651), (2, 110808, 190197)]
    graded_runs = [(s, e + 1) for _, s, e in half_blocks]
    pub_runs = pm.valid_runs(gt_codes)  # [(11185, 18685)]
    pub_start, pub_end = pub_runs[0] if pub_runs else (11185, 18685)

    # Exact 5-minute Ground Truth statistics
    gt_win = gt_codes[pub_start:pub_end]
    n_pub_frames = len(gt_win)
    gt_spells_pub = pm.possession_spells(gt_codes, pub_runs)
    gt_segs_pub = pm.segments(gt_codes, pub_runs)
    gt_trans_pub = pm.transitions(gt_codes, pub_runs)
    gt_durs_pub = [(b - a) / fps for a, b, _ in gt_spells_pub]

    gt_cnt_h = int(np.sum(gt_win == pm.H))
    gt_cnt_a = int(np.sum(gt_win == pm.A))
    gt_cnt_d = int(np.sum(gt_win == pm.D))
    gt_alive = max(1, gt_cnt_h + gt_cnt_a)

    gt_summary_5m = {
        "start": int(pub_start),
        "end": int(pub_end),
        "n_frames": int(n_pub_frames),
        "total_s": n_pub_frames / fps,
        "n_spells": len(gt_spells_pub),
        "n_spells_H": sum(1 for _, _, k in gt_spells_pub if k == pm.H),
        "n_spells_A": sum(1 for _, _, k in gt_spells_pub if k == pm.A),
        "n_dead_blocks": sum(1 for _, _, k in gt_segs_pub if k == pm.D),
        "n_segments": len(gt_segs_pub),
        "n_transitions": len(gt_trans_pub),
        "mean_dur_s": float(np.mean(gt_durs_pub)) if gt_durs_pub else 0.0,
        "sec_H": gt_cnt_h / fps,
        "sec_A": gt_cnt_a / fps,
        "sec_D": gt_cnt_d / fps,
        "pct_H": 100.0 * gt_cnt_h / max(1, n_pub_frames),
        "pct_A": 100.0 * gt_cnt_a / max(1, n_pub_frames),
        "pct_D": 100.0 * gt_cnt_d / max(1, n_pub_frames),
        "home_share_pct": 100.0 * gt_cnt_h / gt_alive,
    }

    results_dir = os.path.join(EVAL_DIR, "results")
    csv_paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))

    submissions = []
    skipped = []

    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pm.read_label_csv(path)
            reason = pm.validate_submission(df, n_frames)
            if reason:
                skipped.append({"name": name, "reason": reason})
                continue
            pred_codes = pm.label_codes(df)
        except Exception as exc:
            skipped.append({"name": name, "reason": str(exc)})
            continue

        ev = pm.evaluate(gt_codes, pred_codes)
        full_st = _full_match_stats(pred_codes, graded_runs, fps)

        # Confusion matrix breakdown on 5m graded window (rows=GT H,A,D; cols=Pred H,A,D,X)
        cm = ev["confusion"]
        correct_frames = int(cm[pm.H][pm.H] + cm[pm.A][pm.A] + cm[pm.D][pm.D])
        err_frames = int(n_pub_frames - correct_frames)
        err_swap_ha_frames = int(cm[pm.H][pm.A] + cm[pm.A][pm.H])
        err_dead_alive_frames = int(
            cm[pm.D][pm.H] + cm[pm.D][pm.A] + cm[pm.H][pm.D] + cm[pm.A][pm.D]
        )
        err_abstain_x_frames = int(cm[pm.H][pm.X] + cm[pm.A][pm.X] + cm[pm.D][pm.X])

        # 5-minute spells & transitions on abstention-filled sequence (matching official scorer)
        filled_pub = pm.fill_abstentions(pred_codes, pub_runs)
        pred_spells_5m = pm.possession_spells(filled_pub, pub_runs)
        pred_spells_h_5m = sum(1 for _, _, k in pred_spells_5m if k == pm.H)
        pred_spells_a_5m = sum(1 for _, _, k in pred_spells_5m if k == pm.A)

        pred_win = pred_codes[pub_start:pub_end]
        p_cnt_h = int(np.sum(pred_win == pm.H))
        p_cnt_a = int(np.sum(pred_win == pm.A))
        p_cnt_d = int(np.sum(pred_win == pm.D))
        p_cnt_x = int(np.sum(pred_win == pm.X))

        loose_tau = pm.TAUS[-1]  # 25 frames = +/- 1.0s
        tr = ev["transitions"][loose_tau]
        sp50 = ev["spell_f1"][0.5]

        submissions.append({
            "name": name,
            "is_baseline": name.startswith("baseline_"),
            # Official scores
            "spell_f1_mean": ev["spell_f1_mean"],
            "spell_f1_iou50": ev["spell_f1_at_headline"],
            "macro_f1": ev["macro_f1"],
            "per_class": ev["per_class"],
            # Frame agreement & exact error breakdown on 5m
            "accuracy_pct": 100.0 * correct_frames / max(1, n_pub_frames),
            "correct_frames": correct_frames,
            "correct_s": correct_frames / fps,
            "err_frames": err_frames,
            "err_s": err_frames / fps,
            "err_swap_ha_s": err_swap_ha_frames / fps,
            "err_dead_alive_s": err_dead_alive_frames / fps,
            "err_abstain_x_s": err_abstain_x_frames / fps,
            # Actions (spells) on 5m & deviation from GT
            "n_spells_5m": ev["n_pred_spells"],
            "n_spells_H_5m": pred_spells_h_5m,
            "n_spells_A_5m": pred_spells_a_5m,
            "diff_spells_5m": int(ev["n_pred_spells"] - gt_summary_5m["n_spells"]),
            "spells_matched_50": int(sp50["n_matched"]),
            "spells_missed_50": int(sp50["n_missed"]),
            "spells_spurious_50": int(sp50["n_spurious"]),
            "mean_dur_5m_s": ev["mean_spell_duration_s"],
            "diff_dur_5m_s": float(ev["mean_spell_duration_s"] - gt_summary_5m["mean_dur_s"]),
            # Transitions (label changes) on 5m & deviation from GT
            "n_trans_5m": int(tr["n_pred"]),
            "diff_trans_5m": int(tr["n_pred"] - gt_summary_5m["n_transitions"]),
            "trans_matched_1s": int(tr["n_matched"]),
            "trans_missed_1s": int(tr["n_missed"]),
            "trans_spurious_1s": int(tr["n_spurious"]),
            "mean_latency_frames": tr["mean_latency"],
            # Seconds per label on 5m & deviation from GT
            "sec_H_5m": p_cnt_h / fps,
            "sec_A_5m": p_cnt_a / fps,
            "sec_D_5m": p_cnt_d / fps,
            "sec_X_5m": p_cnt_x / fps,
            "diff_sec_H_5m": (p_cnt_h - gt_cnt_h) / fps,
            "diff_sec_A_5m": (p_cnt_a - gt_cnt_a) / fps,
            "diff_sec_D_5m": (p_cnt_d - gt_cnt_d) / fps,
            "pred_home_share_5m": ev["pred_home_share"] * 100.0,
            "diff_home_share_5m": (ev["pred_home_share"] * 100.0) - gt_summary_5m["home_share_pct"],
            # Full match 90m stats + RLE
            "full_match": full_st,
            "rle": _rle_encode(pred_codes),
        })

    # Rank submissions by official rule: spell_f1_mean desc, then macro_f1 desc
    submissions.sort(key=lambda s: (-s["spell_f1_mean"], -s["macro_f1"], s["name"]))
    for idx, sub in enumerate(submissions, start=1):
        sub["rank"] = idx

    return {
        "game": GAME,
        "n_frames": n_frames,
        "fps": fps,
        "public_window": gt_summary_5m,
        "halves": [
            {"half": h, "start": int(s), "end": int(e)}
            for h, s, e in half_blocks
        ],
        "gt_rle": _rle_encode(gt_codes),
        "submissions": submissions,
        "skipped": skipped,
    }


# ------------------------------------------------------------------------------------------------
# Seconda pagina: le 10 partite esterne (7 DFL + 3 Metrica), sporcate come il Barça
# ------------------------------------------------------------------------------------------------
# La soluzione vera di ogni partita sta in data/altre_partite/<fonte>_pulite/<id>/<id>_possession_gt.csv
# (le etichette originali, mai corrette). Le risposte dei modelli stanno in
# data/risultati_esterne/<modello>/<id>.csv, nello stesso formato del file del Barça (frame,label), e
# devono essere previste SENZA aver visto quella partita in allenamento. Tutto resta dentro data/,
# quindi GitHub non lo vede.

DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
EXT_MATCH_DIR = os.path.join(DATA_DIR, "altre_partite")
EXT_RESULTS_DIR = os.path.join(DATA_DIR, "risultati_esterne")
EXT_SOURCES = (("dfl", "DFL"), ("metrica", "Metrica"))
EXT_FPS = 25.0


def _ext_matches() -> list:
    """[(fonte, nome fonte, id, csv della soluzione, xml)] delle partite esterne con la soluzione."""
    out = []
    for src, label in EXT_SOURCES:
        for folder in sorted(glob.glob(os.path.join(EXT_MATCH_DIR, src + "_pulite", "*"))):
            mid = os.path.basename(folder)
            gt = os.path.join(folder, f"{mid}_possession_gt.csv")
            if os.path.isfile(gt):
                out.append((src, label, mid, gt, os.path.join(folder, f"{mid}_atd.xml")))
    return out


def _ext_halves(xml_path: str, n_frames: int) -> list:
    """[(tempo, primo frame, ultimo frame)] dai parametri dell'XML (che contano i frame da 1)."""
    params = {}
    try:
        for el in ET.parse(xml_path).getroot().iter():
            if el.tag.split("}")[-1] == "ProviderParameter":
                params.setdefault(el.findtext("Name"), el.findtext("Value"))
        halves = []
        for h, key in ((1, "first_half"), (2, "second_half")):
            if params.get(key + "_start") and params.get(key + "_end"):
                halves.append((h, int(params[key + "_start"]) - 1, int(params[key + "_end"]) - 1))
        if halves:
            return halves
    except (OSError, ET.ParseError, ValueError):
        pass
    return [(1, 0, n_frames - 1)]


def _ext_teams(xml_path: str) -> tuple:
    """(casa, ospiti) dall'XML; None se i nomi sono generici (le partite Metrica dicono solo Home/Away)."""
    try:
        names = [el.findtext("Name") for el in ET.parse(xml_path).getroot().iter()
                 if el.tag.split("}")[-1] == "Team"]
    except (OSError, ET.ParseError):
        names = []
    if len(names) >= 2 and not ({(names[0] or "").lower(), (names[1] or "").lower()} <= {"home", "away", ""}):
        return names[0], names[1]
    return None, None


def _with_halftime_gap(codes: np.ndarray, halves: list) -> np.ndarray:
    """Un frame X tra i due tempi: nessun possesso può scavalcare l'intervallo, come nel Barça."""
    if len(halves) < 2:
        return codes
    cut = halves[1][1]
    return np.concatenate([codes[:cut], np.array([pm.X], dtype=codes.dtype), codes[cut:]])


def _ext_compare(gt_codes: np.ndarray, pred_codes: np.ndarray, halves: list, fps: float) -> dict:
    """Gli stessi numeri della tabella del Barça, ma sull'intera partita."""
    g = _with_halftime_gap(gt_codes, halves)
    p = _with_halftime_gap(pred_codes, halves)
    ev = pm.evaluate(g, p)
    cm = ev["confusion"]
    graded = int(np.sum(g != pm.X))
    correct = int(cm[pm.H][pm.H] + cm[pm.A][pm.A] + cm[pm.D][pm.D])
    sp50 = ev["spell_f1"][0.5]
    tr = ev["transitions"][pm.TAUS[-1]]
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    graded_mask = g != pm.X
    sec = {k: float(np.sum(filled[graded_mask] == c)) / fps for k, c in (("H", pm.H), ("A", pm.A), ("D", pm.D))}
    return {
        "spell_f1_mean": ev["spell_f1_mean"],
        "spell_f1_iou50": ev["spell_f1_at_headline"],
        "macro_f1": ev["macro_f1"],
        "f1_D": ev["per_class"]["D"]["f1"],
        "accuracy_pct": 100.0 * correct / max(1, graded),
        "err_s": (graded - correct) / fps,
        "err_swap_ha_s": (cm[pm.H][pm.A] + cm[pm.A][pm.H]) / fps,
        "err_dead_alive_s": (cm[pm.D][pm.H] + cm[pm.D][pm.A] + cm[pm.H][pm.D] + cm[pm.A][pm.D]) / fps,
        "n_spells": ev["n_pred_spells"],
        "spells_matched_50": int(sp50["n_matched"]),
        "spells_spurious_50": int(sp50["n_spurious"]),
        "mean_dur_s": ev["mean_spell_duration_s"],
        "n_trans": int(tr["n_pred"]),
        "trans_matched_1s": int(tr["n_matched"]),
        "trans_spurious_1s": int(tr["n_spurious"]),
        "sec_H": sec["H"], "sec_A": sec["A"], "sec_D": sec["D"],
    }


def _ext_gt_summary(gt_codes: np.ndarray, halves: list, fps: float) -> dict:
    g = _with_halftime_gap(gt_codes, halves)
    runs = pm.valid_runs(g)
    spells = pm.possession_spells(g, runs)
    segs = pm.segments(g, runs)
    durs = [(b - a) / fps for a, b, _ in spells]
    graded = g != pm.X
    return {
        "graded_s": float(np.sum(graded)) / fps,
        "n_spells": len(spells),
        "n_spells_H": sum(1 for _, _, k in spells if k == pm.H),
        "n_spells_A": sum(1 for _, _, k in spells if k == pm.A),
        "n_dead_blocks": sum(1 for _, _, k in segs if k == pm.D),
        "n_trans": len(pm.transitions(g, runs)),
        "mean_dur_s": float(np.mean(durs)) if durs else 0.0,
        "sec_H": float(np.sum(g[graded] == pm.H)) / fps,
        "sec_A": float(np.sum(g[graded] == pm.A)) / fps,
        "sec_D": float(np.sum(g[graded] == pm.D)) / fps,
    }


def build_external_payload() -> dict:
    """Soluzioni delle 10 partite esterne e risposte di ogni modello in data/risultati_esterne/."""
    models = sorted(
        d for d in os.listdir(EXT_RESULTS_DIR) if os.path.isdir(os.path.join(EXT_RESULTS_DIR, d))
    ) if os.path.isdir(EXT_RESULTS_DIR) else []
    matches, skipped = [], []
    per_model = {m: [] for m in models}
    for src, src_label, mid, gt_path, xml_path in _ext_matches():
        gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
        n = len(gt_codes)
        halves = _ext_halves(xml_path, n)
        home, away = _ext_teams(xml_path)
        subs = []
        for m in models:
            path = os.path.join(EXT_RESULTS_DIR, m, f"{mid}.csv")
            if not os.path.isfile(path):
                continue
            try:
                df = pm.read_label_csv(path)
                reason = pm.validate_submission(df, n)
                if reason:
                    skipped.append({"name": f"{m}/{mid}", "reason": reason})
                    continue
                pred = pm.label_codes(df)
            except Exception as exc:  # un file rotto non deve fermare la pagina
                skipped.append({"name": f"{m}/{mid}", "reason": str(exc)})
                continue
            row = _ext_compare(gt_codes, pred, halves, EXT_FPS)
            row.update({"name": m, "is_baseline": m.startswith("baseline_"), "rle": _rle_encode(pred)})
            subs.append(row)
            per_model[m].append((src, mid, row["spell_f1_mean"], row["macro_f1"]))
        subs.sort(key=lambda s: (-s["spell_f1_mean"], -s["macro_f1"], s["name"]))
        for i, s in enumerate(subs, start=1):
            s["rank"] = i
        matches.append({
            "id": mid, "source": src, "source_label": src_label, "n_frames": n, "fps": EXT_FPS,
            "home": home, "away": away,
            "halves": [{"half": h, "start": s, "end": e} for h, s, e in halves],
            "gt": _ext_gt_summary(gt_codes, halves, EXT_FPS),
            "gt_rle": _rle_encode(gt_codes),
            "submissions": subs,
        })

    n_matches = len(matches)
    best = {}
    for mt in matches:
        if mt["submissions"]:
            top = mt["submissions"][0]["spell_f1_mean"]
            for s in mt["submissions"]:
                if s["spell_f1_mean"] >= top - 1e-12:
                    best[s["name"]] = best.get(s["name"], 0) + 1
    summary = []
    for m, rows in per_model.items():
        if not rows:
            continue
        by_src = {}
        for src, _, sp, _ in rows:
            by_src.setdefault(src, []).append(sp)
        summary.append({
            "name": m,
            "is_baseline": m.startswith("baseline_"),
            "n_matches": len(rows),
            "complete": len(rows) == n_matches,
            "spell_f1_mean": float(np.mean([r[2] for r in rows])),
            "spell_f1_worst": float(np.min([r[2] for r in rows])),
            "spell_f1_best": float(np.max([r[2] for r in rows])),
            "macro_f1": float(np.mean([r[3] for r in rows])),
            "by_source": {src: float(np.mean(v)) for src, v in by_src.items()},
            "best_in": best.get(m, 0),
            "per_match": {mid: sp for _, mid, sp, _ in rows},
        })
    summary.sort(key=lambda s: (not s["complete"], -s["spell_f1_mean"], s["name"]))
    for i, s in enumerate(summary, start=1):
        s["rank"] = i
    return {
        "matches": matches,
        "summary": summary,
        "sources": [{"id": s, "label": l} for s, l in EXT_SOURCES],
        "skipped": skipped,
        "results_dir": os.path.relpath(EXT_RESULTS_DIR, ROOT) if EXT_RESULTS_DIR.startswith(ROOT) else EXT_RESULTS_DIR,
    }


class ExternalCache:
    data_bytes: bytes = b""
    signature: tuple = ()
    lock = threading.Lock()

    @staticmethod
    def _current_signature() -> tuple:
        paths = sorted(glob.glob(os.path.join(EXT_RESULTS_DIR, "*", "*.csv")))
        return tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths)

    @classmethod
    def refresh(cls) -> bytes:
        sig = cls._current_signature()
        raw = json.dumps(build_external_payload(), separators=(",", ":")).encode("utf-8")
        with cls.lock:
            cls.data_bytes = raw
            cls.signature = sig
        return raw

    @classmethod
    def get(cls) -> bytes:
        sig = cls._current_signature()
        with cls.lock:
            if cls.data_bytes and cls.signature == sig:
                return cls.data_bytes
        return cls.refresh()


class DashboardCache:
    data_bytes: bytes = b""
    signature: tuple = ()
    lock = threading.Lock()

    @staticmethod
    def _current_signature() -> tuple:
        results_dir = os.path.join(EVAL_DIR, "results")
        paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
        return tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths)

    @classmethod
    def refresh(cls) -> bytes:
        sig = cls._current_signature()
        payload = build_dashboard_payload()
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with cls.lock:
            cls.data_bytes = raw
            cls.signature = sig
        return raw

    @classmethod
    def get(cls) -> bytes:
        sig = cls._current_signature()
        with cls.lock:
            if cls.data_bytes and cls.signature == sig:
                return cls.data_bytes
        return cls.refresh()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            index_path = os.path.join(HERE, "index.html")
            with open(index_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/data":
            body = DashboardCache.get()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/refresh":
            body = DashboardCache.refresh()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route in ("/esterne", "/esterne.html"):
            with open(os.path.join(HERE, "esterne.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/esterne":
            body = ExternalCache.get()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Visual Comparison Dashboard (Read-Only)")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open web browser")
    args = parser.parse_args()

    print("Caricamento risultati in sola lettura da evaluation/results/ ...", flush=True)
    DashboardCache.refresh()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"\nDashboard attiva su:  {url}", flush=True)
    print("Premi Ctrl+C per chiudere il server.\n", flush=True)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiusura dashboard.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
