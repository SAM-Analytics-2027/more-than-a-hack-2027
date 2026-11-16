"""Reading matches in the format of the competition files (Metrica's FIFA EPTS).

This is the part of data_prep/match_format.py that the models need.

A match is a dictionary, always with the same fields:
    name           short name of the match
    home, away     names of the two teams
    fps            frames per second (25)
    length, width  pitch size in metres
    period         for each frame: 1 or 2 if it is a frame of play, 0 otherwise
    px, py         player positions in metres, shape (frame, slot); NaN when the slot is not tracked
    team           for each slot: 0 = home, 1 = away
    bx, by         ball position in metres; NaN when it is not visible
    labels         'H', 'A', 'D' or 'X' for each frame (external matches only; None for ours)

On disk:
    <name>_atd.txt              one line per frame: number:x,y,speed;...;x,y,speed:x,y,ball speed
    <name>_atd.xml              teams, player slots, pitch, fps, halves
    <name>_possession_gt.csv    frame,label (external matches only)

    read_ours()            Barça–Athletic (data/tactical/)
    read_epts(folder)      one match in that format, for example data/external/dfl_degraded/J03WMX
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

# The repo's data/ folder (the one with tactical/ and external/); it can be changed with MTAH_DATA_DIR
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.environ.get('MTAH_DATA_DIR') or os.path.join(ROOT, 'data')

from constants import (
    FPS, HOME, AWAY, OUR_FRAMES, OUR_LENGTH, OUR_WIDTH, OUR_HALVES,
    BROADCAST_MAX_CENTROID_DIST,
)


def _only(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError('looked for {} in {}: found {} files'.format(pattern, folder, len(hits)))
    return hits[0]


def _fuse_broadcast_ball(tac, data_dir):
    """Fills the missing ball coordinates of the tactical tracking with those of the synchronised broadcast tracking,
    discarding TV replay frames by checking that the players' centroid is consistent.
    """
    bro_dir = os.path.join(data_dir, 'broadcast')
    if not os.environ.get('MV_USE_BROADCAST') or not os.path.isdir(bro_dir):
        return tac
    try:
        bro = read_epts(bro_dir, name='Broadcast', length=OUR_LENGTH, width=OUR_WIDTH)
    except Exception:
        return tac

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        tac_cx = np.nanmean(tac['px'], axis=1)
        tac_cy = np.nanmean(tac['py'], axis=1)
        bro_cx = np.nanmean(bro['px'], axis=1)
        bro_cy = np.nanmean(bro['py'], axis=1)

    for half, a, b in OUR_HALVES:
        tx, ty = tac['bx'][a:b + 1], tac['by'][a:b + 1]
        lo_d = -10000 if half == 1 else -35000
        hi_d = 0 if half == 1 else -25000
        best_d, best_err = 0, 1e9
        for d in range(lo_d, hi_d + 1, 10):
            b_lo, b_hi = a + d, b + d + 1
            if b_lo < 0 or b_hi > len(bro['bx']):
                continue
            bx, by = bro['bx'][b_lo:b_hi], bro['by'][b_lo:b_hi]
            both = (~np.isnan(tx)) & (~np.isnan(bx))
            if int(both.sum()) < 2000:
                continue
            err = float(np.median(np.hypot(tx[both] - bx[both], ty[both] - by[both])))
            if err < best_err:
                best_err, best_d = err, d
        for d in range(best_d - 12, best_d + 13):
            b_lo, b_hi = a + d, b + d + 1
            if b_lo < 0 or b_hi > len(bro['bx']):
                continue
            bx, by = bro['bx'][b_lo:b_hi], bro['by'][b_lo:b_hi]
            both = (~np.isnan(tx)) & (~np.isnan(bx))
            if int(both.sum()) < 2000:
                continue
            err = float(np.median(np.hypot(tx[both] - bx[both], ty[both] - by[both])))
            if err < best_err:
                best_err, best_d = err, d
        if best_err < 2.0:
            t_idx = np.arange(a, b + 1)
            b_idx = t_idx + best_d
            ok_b = (b_idx >= 0) & (b_idx < len(bro['bx']))
            t_idx, b_idx = t_idx[ok_b], b_idx[ok_b]
            miss = np.isnan(tac['bx'][t_idx]) & (~np.isnan(bro['bx'][b_idx]))
            c_dist = np.hypot(tac_cx[t_idx] - bro_cx[b_idx], tac_cy[t_idx] - bro_cy[b_idx])
            live_ok = np.isnan(c_dist) | (c_dist < BROADCAST_MAX_CENTROID_DIST)
            use = miss & live_ok
            tac['bx'][t_idx[use]] = bro['bx'][b_idx[use]]
            tac['by'][t_idx[use]] = bro['by'][b_idx[use]]
    return tac


def read_ours(data_dir=DATA_DIR):
    """Barça–Athletic from the competition files (tactical/ + ball fused from broadcast/)."""
    tac = read_epts(os.path.join(data_dir, 'tactical'), name='Barça–Athletic',
                    length=OUR_LENGTH, width=OUR_WIDTH, halves=OUR_HALVES)
    return _fuse_broadcast_ball(tac, data_dir)


def _xml_parts(path):
    root = ET.parse(path).getroot()
    elements = [(el.tag.split('}')[-1], el) for el in root.iter()]
    params = {}
    for tag, el in elements:
        if tag == 'ProviderParameter':
            params.setdefault(el.findtext('Name'), el.findtext('Value'))
    return elements, params


def read_epts(folder, name=None, length=None, width=None, halves=None, labels_folder=None):
    """One match in the format of the competition files (Metrica's FIFA EPTS): *_atd.txt and *_atd.xml.

    If there is a *_possession_gt.csv (frame,label), the labels are read too: it is looked for in `labels_folder`,
    if given, otherwise in the folder itself. Degraded matches have no CSV: their labels are those of the
    clean version, for example read_epts('.../dfl_degraded/J03WMX', labels_folder='.../dfl_clean/J03WMX').
    The halves come from `halves` ((half, first frame, last frame), ...) if given, otherwise from the
    first/second_half_start/end parameters of the XML (frame numbers of the .txt file, which start at 1).
    """
    xml_path = _only(folder, '*_atd.xml')
    elements, params = _xml_parts(xml_path)
    teams = [el for tag, el in elements if tag == 'Team']
    team = np.array([HOME if el.get('teamId') == teams[0].get('id') else AWAY
                     for tag, el in elements if tag == 'Player'], dtype=np.int8)
    n_slots = len(team)
    spec = next(el for tag, el in elements if tag == 'DataFormatSpecification')
    n_frames = int(spec.get('endFrame'))
    field = next(el for tag, el in elements if tag == 'FieldSize')
    length = length or float(field.findtext('Width'))
    width = width or float(field.findtext('Height'))

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    with open(_only(folder, '*_atd.txt')) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip('\n').split(':')
            row = int(frame_id) - 1  # frame_id starts at 1, video frames at 0
            values = np.fromstring(players.replace(';', ','), sep=',', dtype=np.float32).reshape(n_slots, 3)
            px[row], py[row] = values[:, 0], values[:, 1]  # the third column is the speed: it is recomputed
            ball_values = np.fromstring(ball, sep=',', dtype=np.float32)
            bx[row], by[row] = ball_values[0], ball_values[1]

    period = np.zeros(n_frames, dtype=np.int8)
    if halves is None:
        halves = [(half, int(params[key + '_start']) - 1, int(params[key + '_end']) - 1)
                  for half, key in ((1, 'first_half'), (2, 'second_half')) if params.get(key + '_start')]
    for half, first, last in halves:
        period[first:last + 1] = half

    labels = None
    gt = glob.glob(os.path.join(labels_folder or folder, '*_possession_gt.csv'))
    if gt:
        frames, letters = np.loadtxt(gt[0], delimiter=',', skiprows=1, dtype=str, unpack=True)
        labels = np.full(n_frames, 'X', dtype='<U1')
        labels[frames.astype(int)] = letters
    competition = next((el for tag, el in elements if tag == 'Competition'), None)
    info = {  # match information, to write it back unchanged with save_epts
        'date': next((el.text or '' for tag, el in elements if tag == 'FileDate'), ''),
        'provider': next((el.text or '' for tag, el in elements if tag == 'ProviderName'), ''),
        'source': params.get('tracking_source') or '', 'matchday': params.get('matchday') or '',
        'competition': competition.get('id', '') if competition is not None else '',
        'team_ids': (teams[0].get('id'), teams[1].get('id')),
        'home_attacks': params.get('attack_direction_first_half') or 'left_to_right',
    }
    return {
        'name': name or os.path.basename(os.path.normpath(folder)), 'info': info,
        'home': teams[0].findtext('Name'), 'away': teams[1].findtext('Name'),
        'fps': FPS, 'length': length, 'width': width, 'period': period,
        'px': (px - 0.5) * length, 'py': (py - 0.5) * width, 'team': team,
        'bx': (bx - 0.5) * length, 'by': (by - 0.5) * width, 'labels': labels,
    }
