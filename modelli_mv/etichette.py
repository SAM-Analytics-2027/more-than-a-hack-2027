"""DFL labels without ricochets: brings them closer to the possession definition of the rules (CHALLENGE.md).

DFL changes possession at every touch; the rules say that a touch without control, a deflection or a
ricochet does not change possession. The filter is Stefano's (possession_model/dfl_dataset.py, branch ste),
with his thresholds, rewritten for our files. It uses the CLEAN tracking, where the ball is clearly visible:

- only short possessions (at most 1.6 s) next to other live play are considered;
- if the ball goes between at least two team-mates (a real pass) the possession stays;
- if a single player touches it, under control (ball within 1.6 m and below 16 m/s) for less than 0.24 s
  (0.36 s if a pause follows right after), it is a ricochet: the stretch goes to the previous team
  (or to the neighbour with the longest possession).

It is used for TRAINING only: evaluation always uses the original labels.

    python modelli_mv/etichette.py      saves the labels in data/cache_mv/etichette_senza_rimbalzi/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dati  # noqa: E402
from constants import REACH, PASS_SPEED, MIN_CTRL_S, MAX_PROBE_S  # noqa: E402

OUT_DIR = os.path.join(dati.CACHE, 'etichette_senza_rimbalzi')


def _runs(v):
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    a = np.concatenate(([0], cut))
    b = np.concatenate((cut, [len(v)]))
    return list(zip(a.tolist(), b.tolist(), v[a].tolist()))


def without_ricochets(y, match, fps=25):
    """y: labels 0 H, 1 A, 2 D, 3 X. match: the CLEAN match (formato_comune dictionary)."""
    out = y.copy()
    px, py, bx, by, team = match['px'], match['py'], match['bx'], match['by'], match['team']
    min_ctrl = max(2, int(round(MIN_CTRL_S * fps)))
    min_ctrl_pre_dead = max(3, int(round(1.5 * MIN_CTRL_S * fps)))
    max_probe = int(round(MAX_PROBE_S * fps))
    for half in (1, 2):
        idx = np.flatnonzero(match['period'] == half)
        if len(idx) == 0:
            continue
        s0, e0 = idx[0], idx[-1] + 1
        sub = out[s0:e0].copy()
        changed = True
        while changed:
            changed = False
            runs = _runs(sub)
            for i, (a, b, lab) in enumerate(runs):
                if lab not in (0, 1) or b - a > max_probe:
                    continue
                prev_r = runs[i - 1] if i > 0 else None
                next_r = runs[i + 1] if i + 1 < len(runs) else None
                prev_live = prev_r is not None and prev_r[2] in (0, 1)
                next_live = next_r is not None and next_r[2] in (0, 1)
                if not prev_live and not next_live:
                    continue
                fa, fb = s0 + a, s0 + b
                sx, sy = bx[fa:fb], by[fa:fb]
                vis = np.isfinite(sx)
                ctrl, touchers = 0, 0
                if vis.any():
                    cols = team == lab
                    dd = np.hypot(px[fa:fb][:, cols] - sx[:, None], py[fa:fb][:, cols] - sy[:, None])
                    near = np.isfinite(dd) & (dd <= REACH)
                    touchers = int(near.any(axis=0).sum())
                    dmin = np.where(np.isfinite(dd), dd, 99.0).min(axis=1)
                    if len(sx) >= 3:
                        sp = np.hypot(np.gradient(sx) * fps, np.gradient(sy) * fps)
                    else:
                        sp = np.zeros(len(sx))
                    ctrl = int(((dmin <= REACH) & (np.nan_to_num(sp, nan=99.0) <= PASS_SPEED)).sum())
                if touchers >= 2:
                    continue
                need = min_ctrl_pre_dead if (prev_live and not next_live) else min_ctrl
                if ctrl < need:
                    if prev_live and not next_live:
                        target = prev_r[2]
                    else:
                        nb = [r for r in (prev_r, next_r) if r is not None and r[2] in (0, 1)]
                        target = max(nb, key=lambda r: r[1] - r[0])[2]
                    sub[a:b] = target
                    changed = True
                    break
        out[s0:e0] = sub
    return out


def load(mid):
    return np.load(os.path.join(OUT_DIR, mid + '.npy'))


def _spells(y, period):
    n = 0
    for half in (1, 2):
        v = y[period == half]
        n += sum(1 for a, b, k in _runs(v) if k in (0, 1)) if len(v) else 0
    return n


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    for src, mid in dati.external_matches():
        if src != 'dfl':
            continue
        clean = dati.C.fc.read_epts(os.path.join(dati.BASE, 'dfl_pulite', mid))
        y = dati.load(src, mid)['y'].values.astype(np.int8)
        yf = without_ricochets(y, clean)
        np.save(os.path.join(OUT_DIR, mid + '.npy'), yf)
        per = clean['period']
        print('{}: possessions {} -> {}, frames changed {:.1%}'.format(
            mid, _spells(y, per), _spells(yf, per), (yf != y).mean()), flush=True)
