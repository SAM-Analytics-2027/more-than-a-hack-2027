"""Read-only Visual Comparison WebApp for More than a Hack 2027.

Completely independent from any possession algorithm:
  - Never modifies data/, evaluation/, or algorithm files (100% read-only in RAM).
  - Automatically discovers every *.csv in evaluation/results/ and compares them against
    evaluation/gt/BAR-ATH_possession_gt_public.csv (exact 5-minute solution) and across
    the entire 90-minute match.

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
