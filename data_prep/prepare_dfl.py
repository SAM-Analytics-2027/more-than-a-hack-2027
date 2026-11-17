"""Prepares the 7 DFL/IDSSE matches in the same format as the competition files, with the labels.

Where they come from: the DFL IDSSE dataset (7 German league matches of 2022/23: 2 from the Bundesliga,
5 from the 2. Bundesliga), CC-BY 4.0 licence. Copy on Hugging Face:
https://huggingface.co/datasets/pysport/idsse-data (the official home is figshare; paper:
https://doi.org/10.1038/s41597-025-04505-y).

How to use it (from the repository root):
    python data_prep/prepare_dfl.py            prepares the matches that are not ready yet (downloads only their originals)
    python data_prep/prepare_dfl.py J03WMX     redoes only that match
    python data_prep/degradation.py            then degrades the 7 clean matches (data/external/dfl_degraded/)

What it produces, for each match, in data/external/dfl_clean/<ID>/ (the same format as the competition tactical/):
    <ID>_atd.txt              the tracking: one line per frame, number:x,y,speed;...:x,y,ball speed,
                              coordinates between 0 and 1, speeds in m/s recomputed from the positions
    <ID>_atd.xml              the description: teams, player slots (Track 1, 2, ...), pitch, fps,
                              start and end of the two halves
    <ID>_possession_gt.csv    the ground truth, frame,label, like the organisers' file (frames from 0)

The labels are translated from the two DFL ball fields:
    BallStatus = 0 (ball not in play)                    -> D
    BallStatus = 1 and BallPossession = 1 (home team)    -> H
    BallStatus = 1 and BallPossession = 2 (away team)    -> A
    ball in play without possession                     -> X (frame not used)

This is the clean version: one slot per player for the whole match and the ball always visible (the degradation
comes later). The ball height and the DFL speeds are not kept, since the competition match does not have them.

The downloaded original files stay in data/external/idsse/:
    DFL_02_01_matchinformation_..._<ID>.xml        teams, players, pitch
    DFL_03_02_events_raw_..._<ID>.xml              events: passes, shots, fouls...
    DFL_04_03_positions_raw_observed_..._<ID>.xml  tracking at 25 fps, with possession and ball alive/dead
The XML files are read with kloppy (only_alive=False, i.e. keeping the dead ball too).
"""
import glob
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

import match_format as fc

REPO = 'https://huggingface.co/datasets/pysport/idsse-data'
LISTING = 'https://huggingface.co/api/datasets/pysport/idsse-data/tree/main?recursive=true'
BASE = os.path.join(fc.DATA_DIR, 'external')
ORIGINALS = os.path.join(BASE, 'idsse')        # files downloaded from the DFL
READY = os.path.join(BASE, 'dfl_clean')        # clean versions, in the format of the competition files


# --------------------------------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------------------------------

def files_in_repo():
    """(path in the repo, size in bytes) of all the XML files of the matches."""
    with urllib.request.urlopen(LISTING, timeout=60) as response:
        items = json.load(response)
    return sorted((i['path'], (i.get('lfs') or {}).get('size') or i.get('size'))
                  for i in items if i.get('type') == 'file' and 'DFL-MAT-' in i['path'] and i['path'].endswith('.xml'))


def download(path, size):
    """Downloads a file, if it is missing or incomplete. First as .part, renamed only when the download is complete."""
    target = os.path.join(ORIGINALS, os.path.basename(path))
    if os.path.exists(target) and os.path.getsize(target) == size:
        return
    started = time.time()
    partial = target + '.part'
    with urllib.request.urlopen('{}/resolve/main/{}'.format(REPO, path), timeout=120) as response, \
            open(partial, 'wb') as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if os.path.getsize(partial) != size:
        raise IOError('{}: downloaded {} bytes instead of {}'.format(path, os.path.getsize(partial), size))
    os.replace(partial, target)
    print('downloaded {} ({:.0f} MB in {:.0f} s)'.format(os.path.basename(path), size / 1e6, time.time() - started))


# --------------------------------------------------------------------------------------------------
# 2. Prepare the clean version
# --------------------------------------------------------------------------------------------------

def only(pattern):
    hits = glob.glob(os.path.join(ORIGINALS, pattern))
    if len(hits) != 1:
        raise FileNotFoundError('looked for {} in {}: found {} files'.format(pattern, ORIGINALS, len(hits)))
    return hits[0]


def match_info(meta_path):
    """Date, competition and matchday from the DFL match information file."""
    general = next(el for el in ET.parse(meta_path).getroot().iter() if el.tag.endswith('General'))
    return {'date': general.get('KickoffTime', ''), 'competition': general.get('CompetitionName', ''),
            'matchday': general.get('MatchDay', ''), 'provider': 'DFL (Sportec Solutions)',
            # written verbatim into the XML of the clean and degraded files, which are fingerprinted in
            # models/frozen/expected_data_hashes.json: changing it would change those files
            'source': 'IDSSE, licenza CC-BY 4.0'}


def convert(match_id):
    """Reads the XML files of a match with kloppy and writes the clean version in the format of the competition files."""
    from kloppy import sportec

    started = time.time()
    meta_path = only('DFL_02_01_matchinformation_*_DFL-MAT-{}.xml'.format(match_id))
    raw_path = only('DFL_04_03_positions_raw_observed_*_DFL-MAT-{}.xml'.format(match_id))
    dataset = sportec.load_tracking(raw_data=raw_path, meta_data=meta_path, only_alive=False)
    meta = dataset.metadata
    home, away = meta.teams
    length, width = meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width
    df = dataset.to_df(engine='pandas')
    players = [(p.player_id, fc.HOME) for p in home.players] + [(p.player_id, fc.AWAY) for p in away.players]
    players = [(pid, team) for pid, team in players if '{}_x'.format(pid) in df.columns]  # only those who played

    # Coordinates: from 0-1 (kloppy) to metres with the origin at the centre (the common format in memory)
    meters = lambda values, size: (values - 0.5) * size
    column = lambda pid, axis: df['{}_{}'.format(pid, axis)].to_numpy(float)

    # Labels: ball not in play -> D; in play -> team in possession (H or A); in play without possession -> X
    alive = df['ball_state'].astype(str).str.lower().str.contains('alive').to_numpy()
    owner = df['ball_owning_team_id'].astype(str).to_numpy()
    labels = np.where(~alive, 'D', np.where(owner == str(home.team_id), 'H', np.where(owner == str(away.team_id), 'A', 'X')))

    match = {
        'name': match_id, 'home': home.name, 'away': away.name,
        'fps': int(round(meta.frame_rate)), 'length': length, 'width': width,
        'period': df['period_id'].to_numpy(int).astype(np.int8),
        'px': meters(np.stack([column(pid, 'x') for pid, _ in players], axis=1), length),
        'py': meters(np.stack([column(pid, 'y') for pid, _ in players], axis=1), width),
        'team': np.array([team for _, team in players], dtype=np.int8),
        'bx': meters(df['ball_x'].to_numpy(float), length), 'by': meters(df['ball_y'].to_numpy(float), width),
        'labels': labels,
    }

    # Attack direction in the first half: the home team attacks to the right if it stands in the left half
    first = match['period'] == 1
    home_x = np.nanmean(match['px'][first][:, match['team'] == fc.HOME])
    away_x = np.nanmean(match['px'][first][:, match['team'] == fc.AWAY])
    home_attacks = 'left_to_right' if home_x < away_x else 'right_to_left'

    fc.save_epts(match, os.path.join(READY, match_id), match_id, team_ids=(home.team_id, away.team_id),
                 home_attacks=home_attacks, session_info=match_info(meta_path))
    print('ready {}: {} - {}, {} frames, {} players, labels {}, in {:.0f} s'.format(
        match_id, home.name, away.name, len(df), len(players),
        ' '.join('{} {:.1f}%'.format(l, 100 * np.mean(labels == l)) for l in 'HADX'), time.time() - started))


def is_ready(match_id):
    """True if the clean version of the match already exists (the three files)."""
    folder = os.path.join(READY, match_id)
    return all(os.path.exists(os.path.join(folder, '{}_{}'.format(match_id, end)))
               for end in ('atd.txt', 'atd.xml', 'possession_gt.csv'))


if __name__ == '__main__':
    os.makedirs(ORIGINALS, exist_ok=True)
    if len(sys.argv) == 2:
        convert(sys.argv[1])
        sys.exit()
    wanted = files_in_repo()
    ids = sorted({os.path.basename(path).split('DFL-MAT-')[1][:-4] for path, _ in wanted})
    missing = [i for i in ids if not is_ready(i)]
    print('{} matches in the repo; to prepare: {}'.format(len(ids), ', '.join(missing) or 'none'))
    for match_id in missing:  # only the originals of the matches not ready yet are downloaded
        for path, size in wanted:
            if path.endswith('DFL-MAT-{}.xml'.format(match_id)):
                download(path, size)
        convert(match_id)
    print('ready in {}: {}'.format(READY, ', '.join(ids)))
