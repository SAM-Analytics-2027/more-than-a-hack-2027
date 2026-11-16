"""Prepares the SkillCorner matches in the same format as the competition files, with the labels.

Where they come from: SkillCorner Open Data (https://github.com/SkillCorner/opendata), 20 matches of the
Australian league (A-League 2024/25), tracking derived from TV footage, 10 frames per second. MIT licence:
SkillCorner is credited.

How to use it (from the repository root):
    python data_prep/prepare_skillcorner.py            prepares the matches not ready yet, clean and degraded
                                                       (downloads only the missing originals)
    python data_prep/prepare_skillcorner.py 1925299    redoes only that match, clean and degraded

The degraded versions, in data/external/skillcorner_degraded/<ID>/, have only <ID>_atd.txt and <ID>_atd.xml (the
labels stay in the clean one), 46 anonymous slots like the competition match and an empty ball speed. They are
obtained with the three pieces of degradation.py and the RECIPES below, with the replays (X) treated as outside play.

What it produces, for each match, in data/external/skillcorner_clean/<ID>/ (the same format as the competition tactical/):
    <ID>_atd.txt              the tracking: one line per frame, number:x,y,speed;...:x,y,ball speed,
                              coordinates between 0 and 1, speeds in m/s recomputed from the positions
    <ID>_atd.xml              the description: teams, player slots, pitch, fps, start and end of the two halves
    <ID>_possession_gt.csv    the ground truth, frame,label, like the organisers' file (frames from 0)

The labels come from the SkillCorner phases of play (<ID>_phases_of_play.csv), which exist only with the ball
in play:
    frame inside a phase of the home / away team                  -> H / A
    frame of play outside every phase                             -> D
    "pause" shorter than 2 s between two possessions, with the
    ball always on the pitch (a hole between two phases, not a
    real pause)                                                   -> the possession stays with the last team
    frame without a view of the pitch (replays, close-ups)        -> X (frame not used), with empty positions
    frame outside the two halves                                  -> X

The tracking goes from 10 to 25 frames per second: the positions in between are interpolated between the two
nearest SkillCorner frames (never across a hole or between halves); half and label come from the nearest
SkillCorner frame, and a boundary frame whose positions touch a frame without a view becomes X too. SkillCorner
has y from bottom to top, the common format from top to bottom: y changes sign. This is the clean version: one
slot per player for the whole match, with the positions given by SkillCorner (seen, or estimated when the player
is just outside the view); no ball height.

The downloaded original files stay in data/external/skillcorner_raw/<ID>/:
    <ID>_match.json                    teams, players, pitch, halves
    <ID>_tracking_extrapolated.jsonl   tracking at 10 fps, with the projection of the camera view
    <ID>_phases_of_play.csv            phases of play: from which frame to which, which team has the ball
    <ID>_dynamic_events.csv            events (player possessions, ...): used only for checks
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

import match_format as fc
import degradation as dg

REPO_RAW = 'https://raw.githubusercontent.com/SkillCorner/opendata/master/data'
REPO_LFS = 'https://media.githubusercontent.com/media/SkillCorner/opendata/master/data'  # large files (Git LFS)
BASE = os.path.join(fc.DATA_DIR, 'external')
ORIGINALS = os.path.join(BASE, 'skillcorner_raw')      # files downloaded from SkillCorner
READY = os.path.join(BASE, 'skillcorner_clean')        # clean versions, in the format of the competition files
DEGRADED = os.path.join(BASE, 'skillcorner_degraded')  # degraded versions, like the competition tracking
FILES = ('match.json', 'tracking_extrapolated.jsonl', 'phases_of_play.csv', 'dynamic_events.csv')
SOURCE_FPS = 10
MAX_GAP = 2.0    # s: a "pause" shorter than this between two possessions, with the ball on the pitch, is a hole between two phases

# Degradation recipes for SkillCorner. Missing ball and players: the same as DFL, which are built on how often
# things happen in the competition match (redoing them on SkillCorner would be biased: the replays break the
# stretches and make them look shorter). Shift: tuned on the 20 SkillCorner matches, where the ball is already
# somewhat shifted because it is estimated from the TV images.
RECIPES = {'ball': dict(dg.BALL),
           'shift': {'amplitude': 0.5, 'timescale': 0.75, 'delay': 1},   # DFL: 0.8, 0.5, 2
           'players': dict(dg.PLAYERS)}
SHIFT_GRID = {'amplitude': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], 'timescale': [0.35, 0.5, 0.75], 'delay': [0, 1, 2]}


# --------------------------------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------------------------------

def download(url, target):
    """Downloads a file, if it is missing. First as .part, renamed only when the download is complete."""
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    started = time.time()
    partial = target + '.part'
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, 'wb') as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    with open(partial, 'rb') as f:
        if f.read(40).startswith(b'version https://git-lfs'):
            raise IOError('{}: received the Git LFS pointer, not the real file'.format(url))
    os.replace(partial, target)
    print('downloaded {} ({:.0f} MB in {:.0f} s)'.format(os.path.basename(target), os.path.getsize(target) / 1e6,
                                                         time.time() - started))


def match_ids():
    """The match ids, from the list in the SkillCorner repo (downloaded to skillcorner_raw/matches.json)."""
    path = os.path.join(ORIGINALS, 'matches.json')
    download(REPO_RAW + '/matches.json', path)
    with open(path, encoding='utf-8') as f:
        return sorted(str(m['id']) for m in json.load(f))


def download_match(match_id):
    for name in FILES:
        base = REPO_LFS if name.endswith('.jsonl') else REPO_RAW
        download('{}/matches/{}/{}_{}'.format(base, match_id, match_id, name),
                 os.path.join(ORIGINALS, match_id, '{}_{}'.format(match_id, name)))


# --------------------------------------------------------------------------------------------------
# 2. Read at 10 fps and label
# --------------------------------------------------------------------------------------------------

def read_source(match_id):
    """Reads a SkillCorner match at 10 fps: positions in the common format (metres, origin at the centre, y
    downwards), halves, camera view, labels."""
    folder = os.path.join(ORIGINALS, match_id)
    path = lambda name: os.path.join(folder, '{}_{}'.format(match_id, name))
    with open(path('match.json'), encoding='utf-8') as f:
        meta = json.load(f)
    home_id, away_id = meta['home_team']['id'], meta['away_team']['id']
    players = {p['id']: p for p in meta['players']}
    with open(path('tracking_extrapolated.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    n = len(rows)
    if any(r['frame'] != i for i, r in enumerate(rows)):
        raise ValueError('{}: the tracking frames are not consecutive from 0'.format(match_id))

    period = np.array([r['period'] or 0 for r in rows], dtype=np.int8)
    camera = np.array([(r.get('image_corners_projection') or {}).get('x_top_left') is not None for r in rows])
    value = lambda v: np.nan if v is None else v
    bx = np.array([value(r['ball_data']['x']) for r in rows], float)
    by = -np.array([value(r['ball_data']['y']) for r in rows], float)

    # One slot per player: first the home team, then the away team, in shirt number order
    seen = {p['player_id'] for r in rows for p in r['player_data']}
    order = sorted(seen, key=lambda pid: (players[pid]['team_id'] != home_id, players[pid].get('number') or 0, pid))
    slot = {pid: k for k, pid in enumerate(order)}
    px = np.full((n, len(order)), np.nan)
    py = np.full((n, len(order)), np.nan)
    for i, r in enumerate(rows):
        for p in r['player_data']:
            px[i, slot[p['player_id']]] = p['x']
            py[i, slot[p['player_id']]] = -p['y']
    team = np.array([fc.HOME if players[pid]['team_id'] == home_id else fc.AWAY for pid in order], dtype=np.int8)

    # Without a camera view nothing is seen: empty positions (they often are already)
    px[~camera], py[~camera], bx[~camera], by[~camera] = np.nan, np.nan, np.nan, np.nan

    # Labels from the phases of play
    phases = pd.read_csv(path('phases_of_play.csv'))
    labels = np.full(n, 'D', dtype='<U1')
    for start, end, owner in zip(phases.frame_start, phases.frame_end, phases.team_in_possession_id):
        labels[int(start):int(end) + 1] = 'H' if owner == home_id else 'A'
    labels = close_gaps(labels, period, bx, by, meta['pitch_length'], meta['pitch_width'])
    labels[period == 0] = 'X'
    labels[(period > 0) & ~camera] = 'X'
    return dict(meta=meta, period=period, camera=camera, px=px, py=py, team=team, bx=bx, by=by, labels=labels)


def close_gaps(labels, period, bx, by, length, width):
    """A "pause" shorter than MAX_GAP between two possessions, with the ball always on the pitch, is not a real
    pause but a hole between two phases of play: the possession stays with the last team (as in the organisers' rules)."""
    labels = labels.copy()
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:] - 1, len(labels) - 1]
    for s, e in zip(starts, ends):
        if labels[s] != 'D' or period[s] == 0 or e - s + 1 >= MAX_GAP * SOURCE_FPS:
            continue
        if s == 0 or e + 1 >= len(labels) or labels[s - 1] not in 'HA' or labels[e + 1] not in 'HA':
            continue
        x, y = bx[s:e + 1], by[s:e + 1]
        ok = np.isfinite(x)
        if not np.any((np.abs(x[ok]) > length / 2) | (np.abs(y[ok]) > width / 2)):
            labels[s:e + 1] = labels[s - 1]
    return labels


# --------------------------------------------------------------------------------------------------
# 3. From 10 to 25 frames per second
# --------------------------------------------------------------------------------------------------

def to_25fps(source):
    """Brings the match from 10 to 25 fps. Positions: interpolated between the two nearest frames of the same half
    (NaN if one of the two is missing). Half, camera view and label: from the nearest frame."""
    n = len(source['period'])
    n_out = (n - 1) * fc.FPS // SOURCE_FPS + 1
    t = np.arange(n_out) * SOURCE_FPS / fc.FPS          # position in the 10 fps tracking
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = t - i0
    w = np.where(source['period'][i0] == source['period'][i1], w, np.round(w))   # between two halves: the nearest
    nearest = np.where(w < 0.5, i0, i1)

    def blend(values):
        a, b = values[i0], values[i1]
        ww = w if values.ndim == 1 else w[:, None]
        out = a + (b - a) * ww
        out = np.where(ww == 0, a, out)
        return np.where(ww == 1, b, out)

    # Camera view: there only if both frames used for the positions have it (otherwise the positions are
    # empty and that boundary frame becomes X too)
    camera = (source['camera'][i0] | (w == 1)) & (source['camera'][i1] | (w == 0))
    period = source['period'][nearest]
    labels = source['labels'][nearest].copy()
    labels[(period > 0) & ~camera] = 'X'
    return dict(source, period=period, camera=camera, labels=labels, px=blend(source['px']), py=blend(source['py']),
                bx=blend(source['bx']), by=blend(source['by']))


# --------------------------------------------------------------------------------------------------
# 4. Save the clean version
# --------------------------------------------------------------------------------------------------

def convert(match_id):
    started = time.time()
    m = to_25fps(read_source(match_id))
    meta = m['meta']
    match = {'name': match_id, 'home': meta['home_team']['name'], 'away': meta['away_team']['name'],
             'fps': fc.FPS, 'length': float(meta['pitch_length']), 'width': float(meta['pitch_width']),
             'period': m['period'], 'px': m['px'], 'py': m['py'], 'team': m['team'],
             'bx': m['bx'], 'by': m['by'], 'labels': m['labels']}

    # Attack direction in the first half, as for DFL: the team standing in the left half attacks to the right
    first = match['period'] == 1
    home_x = np.nanmean(match['px'][first][:, match['team'] == fc.HOME])
    away_x = np.nanmean(match['px'][first][:, match['team'] == fc.AWAY])
    home_attacks = 'left_to_right' if home_x < away_x else 'right_to_left'

    info = {'date': meta.get('date_time', ''), 'provider': 'SkillCorner',
            # written verbatim into the XML of the clean and degraded files, which are fingerprinted in
            # models/frozen/expected_data_hashes.json: changing it would change those files
            'source': 'SkillCorner Open Data (github.com/SkillCorner/opendata), licenza MIT',
            'competition': (meta.get('competition_edition') or {}).get('name', ''),
            'matchday': str((meta.get('competition_round') or {}).get('round_number', ''))}
    fc.save_epts(match, os.path.join(READY, match_id), match_id,
                 team_ids=(meta['home_team']['id'], meta['away_team']['id']),
                 home_attacks=home_attacks, session_info=info)
    labels = match['labels']
    print('ready {}: {} - {}, {} frames, {} players, labels {}, in {:.0f} s'.format(
        match_id, match['home'], match['away'], len(labels), match['px'].shape[1],
        ' '.join('{} {:.1f}%'.format(l, 100 * np.mean(labels == l)) for l in 'HADX'), time.time() - started))


def is_ready(match_id, folder=READY, ends=('atd.txt', 'atd.xml', 'possession_gt.csv')):
    """True if the version of the match already exists (clean: three files; degraded: two)."""
    return all(os.path.exists(os.path.join(folder, match_id, '{}_{}'.format(match_id, end))) for end in ends)


# --------------------------------------------------------------------------------------------------
# 5. Degrade: the same three pieces of degradation.py, with the replays (X) treated as outside play
# --------------------------------------------------------------------------------------------------

def without_x(match):
    """Copy in which the X frames are outside play (period 0), like half-time: this way the measures and the
    degradation pieces do not cross the replays, which have no data."""
    return dict(match, period=np.where(match['labels'] == 'X', 0, match['period']).astype(np.int8))


def pause_labels(labels):
    """The X frames without a camera view next to a pause become D, only to decide where to remove the ball: this
    way a pause with a replay in the middle counts as a single pause, as in the competition match."""
    out = labels.copy()
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:] - 1, len(labels) - 1]
    for s, e in zip(starts, ends):
        if labels[s] == 'X' and ((s > 0 and labels[s - 1] == 'D') or (e + 1 < len(labels) and labels[e + 1] == 'D')):
            out[s:e + 1] = 'D'
    return out


def degrade(clean, pieces=dg.PIECES, recipes=None):
    """The three pieces of degradation.py in order, with the SkillCorner recipes. The labels and the halves stay
    those of the clean version."""
    recipes = dict(RECIPES, **(recipes or {}))
    match = dict(clean)
    if 'ball' in pieces:
        match = dg.remove_ball(dict(match, labels=pause_labels(clean['labels'])), recipes['ball'])
    match = without_x(dict(match, labels=clean['labels']))
    if 'shift' in pieces:
        match = dg.shift_ball(match, recipes['shift'])
    if 'players' in pieces:
        match = dg.remove_players(match, recipes['players'])
    return dict(match, period=clean['period'], labels=clean['labels'])


def save_degraded(match_id):
    """Reads the clean version and saves the degraded one: only atd.txt and atd.xml (the labels stay in the clean
    one), with an empty ball speed, as in the competition file."""
    started = time.time()
    clean = fc.read_epts(os.path.join(READY, match_id))
    dirty = degrade(clean)
    info = clean['info']
    fc.save_epts(dict(dirty, labels=None), os.path.join(DEGRADED, match_id), match_id, team_ids=info['team_ids'],
                 home_attacks=info['home_attacks'], session_info=info, ball_speed=False)
    print('degraded {}: {} slots, in {:.0f} s'.format(match_id, dirty['px'].shape[1], time.time() - started))


if __name__ == '__main__':
    os.makedirs(ORIGINALS, exist_ok=True)
    if len(sys.argv) == 2:
        download_match(sys.argv[1])
        convert(sys.argv[1])
        save_degraded(sys.argv[1])
        sys.exit()
    ids = match_ids()
    missing = [i for i in ids if not is_ready(i)]
    print('{} matches in the repo; clean ones to prepare: {}'.format(len(ids), ', '.join(missing) or 'none'))
    for match_id in missing:  # only the originals of the matches not ready yet are downloaded
        download_match(match_id)
        convert(match_id)
    print('clean versions ready in {}'.format(READY))
    missing = [i for i in ids if not is_ready(i, DEGRADED, ('atd.txt', 'atd.xml'))]
    print('degraded ones to prepare: {}'.format(', '.join(missing) or 'none'))
    for match_id in missing:
        save_degraded(match_id)
    print('degraded versions ready in {}'.format(DEGRADED))
