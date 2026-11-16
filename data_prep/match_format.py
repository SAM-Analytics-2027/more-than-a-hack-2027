"""Common match format and statistics to compare matches.

A match is a dictionary, always with the same fields:
    name           short name of the match
    home, away     names of the two teams
    fps            frames per second (25)
    length, width  pitch size in metres
    period         for each frame: 1 or 2 if it is a frame of play, 0 otherwise
    px, py         player positions in metres, shape (frame, slot); NaN when the slot is not tracked
    team           for each slot: 0 = home, 1 = away
    bx, by         ball position in metres; NaN when it is not visible
    labels         'H', 'A', 'D' or 'X' for each frame (external matches only; None for the competition match)

Coordinates: origin at the centre of the pitch, x along the pitch, y across it (top to bottom,
as in Metrica and kloppy). Speeds are not stored: they are recomputed from the positions, so they are
the same for every source.

On disk every match is stored in the same format as the competition files (Metrica's FIFA EPTS):
    <name>_atd.txt              one line per frame: number:x,y,speed;...;x,y,speed:x,y,ball speed
    <name>_atd.xml              teams, player slots, pitch, fps, halves
    <name>_possession_gt.csv    frame,label (external matches only, like the organisers' ground truth)

    read_ours()            the competition match (tactical/)
    read_epts(folder)      one match in that format, for example data/external/dfl_clean/J03WMX
    save_epts(...)         writes a match in that format (used by prepare_dfl.py and prepare_skillcorner.py)
"""
import glob
import os
import types
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

# The repo's data/ folder (the one with tactical/ and external/); it can be changed with the MTAH_DATA_DIR
# environment variable
DATA_DIR = os.environ.get('MTAH_DATA_DIR') or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SCORER_URL = ('https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/HEAD/'
              'evaluation/possession_metrics.py')
FPS = 25
HOME, AWAY = 0, 1

# Competition match: constants of the official repo (games.py and DATA.md)
OUR_FRAMES = 199319
OUR_LENGTH, OUR_WIDTH = 105.3, 68.0
OUR_HALVES = ((1, 11185, 90651), (2, 110808, 190197))  # graded frames, ends included

# Parameters of the statistics
OUT_MARGIN = 0.5    # metres beyond the line to call the ball "out"
SPEED_STEP = 5      # frames over which speed is measured (0.2 s): reduces the jitter of the positions
BASELINE_WINDOWS = (('raw', 0), ('±2 s', 2), ('±5 s', 5))


# --------------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------------

def _only(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError('looked for {} in {}: found {} files'.format(pattern, folder, len(hits)))
    return hits[0]


def read_ours(data_dir=DATA_DIR):
    """Barça–Athletic from the competition files (tactical/). Pitch and halves from the constants of the official repo."""
    return read_epts(os.path.join(data_dir, 'tactical'), name='Barça–Athletic',
                     length=OUR_LENGTH, width=OUR_WIDTH, halves=OUR_HALVES)


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


def _speeds(x, y, period, fps):
    # Speed (m/s) for the third field of the files: difference over ±2 frames, only within the same half
    out = np.full(x.shape, np.nan, dtype=np.float64)
    step = 2
    same = np.zeros(len(period), dtype=bool)
    same[step:-step] = (period[:-2 * step] == period[2 * step:]) & (period[step:-step] > 0)
    d = np.hypot(x[2 * step:] - x[:-2 * step], y[2 * step:] - y[:-2 * step]) * fps / (2 * step)
    out[step:-step] = d
    out[~same] = np.nan
    return out


_PLAYER_XML = """      <Player id="{k}" teamId="{team_id}">
        <Name>Track {k}</Name>
        <ShirtNumber>{k}</ShirtNumber>
        <ProviderPlayerParameters>
          <ProviderParameter>
            <Name>position_type</Name>
            <Value>Unknown</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_index</Name>
            <Value>-1</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_x</Name>
            <Value>0.0</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_y</Name>
            <Value>0.0</Value>
          </ProviderParameter>
        </ProviderPlayerParameters>
      </Player>
"""

_TEAM_XML = """      <Team id="{team_id}">
        <Name>{name}</Name>
        <ProviderTeamsParameters>
          <ProviderParameter>
            <Name>attack_direction_first_half</Name>
            <Value>{direction}</Value>
          </ProviderParameter>
        </ProviderTeamsParameters>
      </Team>
"""

_PARAM_XML = """        <ProviderParameter>
          <Name>{name}</Name>
          {value}
        </ProviderParameter>
"""


def save_epts(match, folder, prefix, team_ids=('HOME', 'AWAY'), home_attacks='left_to_right', session_info=None,
              ball_speed=True):
    """Writes a match in the same format as the competition files: <prefix>_atd.txt and <prefix>_atd.xml,
    plus <prefix>_possession_gt.csv (frame,label) if the match has labels.

    Coordinates between 0 and 1 (as in the competition file), speeds in m/s recomputed from the positions (in
    the competition file too the third field is in m/s, although the XML says km/h), NaN where data is missing.
    ball_speed=False leaves the ball speed empty (NaN), as almost always in the competition file.
    """
    from xml.sax.saxutils import escape
    info = session_info or {}  # optional: date, provider, source, competition, matchday
    os.makedirs(folder, exist_ok=True)
    length, width, fps, period = match['length'], match['width'], match['fps'], match['period']
    n_frames, n_slots = match['px'].shape

    # 1. Tracking: frame_id:x,y,s;x,y,s;...:bx,by,bs
    nx, ny = match['px'] / length + 0.5, match['py'] / width + 0.5
    ps = np.stack([_speeds(match['px'][:, k], match['py'][:, k], period, fps) for k in range(n_slots)], axis=1)
    ball = np.stack([match['bx'] / length + 0.5, match['by'] / width + 0.5,
                     _speeds(match['bx'], match['by'], period, fps) if ball_speed else np.full(n_frames, np.nan)], axis=1)
    players = np.stack([nx, ny, ps], axis=2).reshape(n_frames, n_slots * 3)
    row_format = ';'.join(['{:.5f},{:.5f},{:.4f}'] * n_slots) + ':{:.5f},{:.5f},{:.4f}'
    with open(os.path.join(folder, prefix + '_atd.txt'), 'w') as out:
        for f in range(n_frames):
            line = '{}:'.format(f + 1) + row_format.format(*players[f], *ball[f])
            out.write(line.replace('nan', 'NaN') + '\n')

    # 2. Description (XML), with the same structure as the competition file
    halves = {}
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        halves[half] = (str(rows[0] + 1), str(rows[-1] + 1)) if len(rows) else ('', '')
    value = lambda v: '<Value>{}</Value>'.format(v) if v != '' else '<Value/>'
    global_params = [('first_half_start', halves[1][0]), ('first_half_end', halves[1][1]),
                     ('second_half_start', halves[2][0]), ('second_half_end', halves[2][1]),
                     ('first_extra_half_start', ''), ('first_extra_half_end', ''),
                     ('second_extra_half_start', ''), ('second_extra_half_end', ''),
                     ('data_type', 'atd'), ('tracking_source', info.get('source', '')), ('version', '1')]
    away_attacks = 'right_to_left' if home_attacks == 'left_to_right' else 'left_to_right'
    xml = ['<?xml version="1.0" encoding="utf-8"?>\n',
           '<main xmlns:tns="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS" '
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="FIFADataTransferFormatEPTSNamespace/'
           'FIFADataTransferFormatEPTS FIFADataTransferFormatEPTS.xsd">\n',
           '  <Metadata>\n    <GlobalConfig>\n',
           '      <FileDate>{}</FileDate>\n'.format(escape(info.get('date', ''))),
           '      <FileName>{}</FileName>\n'.format(escape(prefix)),
           '      <TrackingType>Optical</TrackingType>\n',
           '      <ProviderName>{}</ProviderName>\n'.format(escape(info.get('provider', ''))),
           '      <FrameRate>{:.1f}</FrameRate>\n'.format(fps),
           '      <ProviderGlobalParameters>\n']
    xml += [_PARAM_XML.format(name=k, value=value(v)) for k, v in global_params]
    xml += ['      </ProviderGlobalParameters>\n    </GlobalConfig>\n',
            '    <Sessions>\n      <Session id="session1">\n        <SessionType>Match</SessionType>\n',
            '        <Start>{}</Start>\n'.format(escape(info.get('date', ''))),
            '        <MatchParameters>\n          <Score idLocalTeam="{}" idVisitingTeam="{}">\n'.format(*team_ids),
            '            <LocalTeamScore/>\n            <VisitingTeamScore/>\n          </Score>\n        </MatchParameters>\n',
            '        <FieldSize>\n          <Width>{:g}</Width>\n          <Height>{:g}</Height>\n        </FieldSize>\n'.format(length, width),
            '        <Competition id="{}"/>\n'.format(escape(info.get('competition', ''))),
            '        <ProviderSessionParameters>\n          <ProviderParameter>\n            <Name>matchday</Name>\n'
            '            {}\n          </ProviderParameter>\n        </ProviderSessionParameters>\n'.format(value(info.get('matchday', ''))),
            '      </Session>\n    </Sessions>\n',
            '    <Teams>\n',
            _TEAM_XML.format(team_id=team_ids[0], name=escape(match['home']), direction=home_attacks),
            _TEAM_XML.format(team_id=team_ids[1], name=escape(match['away']), direction=away_attacks),
            '    </Teams>\n    <Players>\n']
    xml += [_PLAYER_XML.format(k=k + 1, team_id=team_ids[match['team'][k]]) for k in range(n_slots)]
    xml += ['    </Players>\n',
            '    <Devices>\n      <Device id="device1">\n        <Name>Device 1</Name>\n        <Sensors>\n'
            '          <Sensor id="position">\n            <Name>Position</Name>\n            <Channels>\n'
            '              <Channel id="x">\n                <Name>position_x</Name>\n                <Unit>normalized</Unit>\n              </Channel>\n'
            '              <Channel id="y">\n                <Name>position_y</Name>\n                <Unit>normalized</Unit>\n              </Channel>\n'
            '            </Channels>\n          </Sensor>\n'
            '          <Sensor id="speed">\n            <Name>Speed in kilometers per hour</Name>\n            <Channels>\n'
            '              <Channel id="s">\n                <Name>speed_kph</Name>\n                <Unit>kph</Unit>\n              </Channel>\n'
            '            </Channels>\n          </Sensor>\n        </Sensors>\n      </Device>\n    </Devices>\n',
            '    <PlayerChannels>\n']
    for k in range(1, n_slots + 1):
        xml += ['      <PlayerChannel channelId="{0}" id="player{1}_{2}" playerId="{1}"/>\n'.format(c, k, n)
                for c, n in (('x', 'x'), ('y', 'y'), ('s', 'speed'))]
    xml += ['    </PlayerChannels>\n  </Metadata>\n  <DataFormatSpecifications>\n',
            '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">\n'.format(n_frames),
            '      <StringRegister name="frameCount"/>\n      <SplitRegister separator=";">\n']
    for k in range(1, n_slots + 1):
        xml += ['        <SplitRegister separator=",">\n'] + \
               ['          <PlayerChannelRef playerChannelId="player{}_{}"/>\n'.format(k, n) for n in ('x', 'y', 'speed')] + \
               ['        </SplitRegister>\n']
    xml += ['      </SplitRegister>\n      <SplitRegister separator=",">\n',
            '        <BallChannelRef channelId="x"/>\n        <BallChannelRef channelId="y"/>\n        <BallChannelRef channelId="s"/>\n',
            '      </SplitRegister>\n    </DataFormatSpecification>\n  </DataFormatSpecifications>\n</main>\n']
    with open(os.path.join(folder, prefix + '_atd.xml'), 'w', encoding='utf-8') as out:
        out.write(''.join(xml))

    # 3. Labels, like the organisers' ground-truth file: frame,label (frames from 0)
    if match.get('labels') is not None:
        with open(os.path.join(folder, prefix + '_possession_gt.csv'), 'w') as out:
            out.write('frame,label\n')
            out.writelines('{},{}\n'.format(f, l) for f, l in enumerate(match['labels']))


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------

def runs(mask):
    """Starts and lengths of the consecutive True stretches."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def runs_in_periods(match, mask):
    """Like runs, but split by half and only on the frames of play: starts and lengths.

    Frames outside play in the middle of a half (for example the SkillCorner frames without a view of the pitch,
    set outside play for the measures) break the stretches. In matches without holes inside the halves nothing changes.
    """
    starts, lengths = [], []
    for p in np.unique(match['period'][match['period'] > 0]):
        idx = np.flatnonzero(match['period'] == p)
        inside = match['period'][idx[0]:idx[-1] + 1] == p
        s, n = runs(np.asarray(mask[idx[0]:idx[-1] + 1], dtype=bool) & inside)
        starts.append(s + idx[0])
        lengths.append(n)
    return np.concatenate(starts), np.concatenate(lengths)


def same_period_ahead(match, step):
    """True where frame i and frame i+step are frames of play in the same half."""
    p = match['period']
    ok = np.zeros(len(p), dtype=bool)
    ok[:-step] = (p[:-step] > 0) & (p[:-step] == p[step:])
    return ok


def player_speeds(match, step=SPEED_STEP):
    """Speed of each slot (m/s) between frame i and frame i+step, NaN if one of the two is missing."""
    px, py = match['px'], match['py']
    out = np.full(px.shape, np.nan, dtype=np.float32)
    out[:-step] = np.hypot(px[step:] - px[:-step], py[step:] - py[:-step]) * match['fps'] / step
    out[~same_period_ahead(match, step)] = np.nan
    return out


def ball_speed(match, step=SPEED_STEP):
    bx, by = match['bx'], match['by']
    out = np.full(len(bx), np.nan, dtype=np.float32)
    out[:-step] = np.hypot(bx[step:] - bx[:-step], by[step:] - by[:-step]) * match['fps'] / step
    out[~same_period_ahead(match, step)] = np.nan
    return out


def nearest(match):
    """Team of the player closest to the ball (-1 if the ball is not visible) and minimum distance."""
    d = np.hypot(match['px'] - match['bx'][:, None], match['py'] - match['by'][:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    d_home = d[:, match['team'] == HOME].min(1)
    d_away = d[:, match['team'] == AWAY].min(1)
    d_min = np.minimum(d_home, d_away)
    ok = np.isfinite(d_min) & (match['period'] > 0)
    team = np.where(ok, np.where(d_home <= d_away, HOME, AWAY), -1)
    return team, np.where(ok, d_min, np.nan)


def baseline(match, codes, seconds):
    """Official baseline, half by half: majority vote over ±seconds, D where there is no vote."""
    out = np.full(len(codes), 'X', dtype='<U1')
    window = int(round(seconds * match['fps']))
    for p in np.unique(match['period'][match['period'] > 0]):
        idx = np.flatnonzero(match['period'] == p)
        c = codes[idx]
        if window == 0:
            out[idx] = np.where(c == HOME, 'H', np.where(c == AWAY, 'A', 'D'))
            continue
        home = np.concatenate(([0], np.cumsum(c == HOME)))
        away = np.concatenate(([0], np.cumsum(c == AWAY)))
        i = np.arange(len(c))
        lo, hi = np.maximum(0, i - window), np.minimum(len(c), i + window + 1)
        h, a = home[hi] - home[lo], away[hi] - away[lo]
        out[idx] = np.where(h + a == 0, 'D', np.where(h >= a, 'H', 'A'))
    return out


def possessions(match, labels):
    """Durations (s) of the possessions: continuous stretches of H or of A within each half."""
    out = []
    for team in ('H', 'A'):
        _, lengths = runs_in_periods(match, labels == team)
        out.append(lengths)
    return np.concatenate(out) / match['fps']


_scorer = None


def scorer():
    """The organisers' official script (possession_metrics.py), downloaded from the official repo."""
    global _scorer
    if _scorer is None:
        with urllib.request.urlopen(SCORER_URL, timeout=30) as response:
            code = response.read().decode('utf-8')
        module = types.ModuleType('possession_metrics')
        exec(compile(code, SCORER_URL, 'exec'), module.__dict__)
        _scorer = module
    return _scorer


def game_sequence(match, labels):
    """The letters of the frames of play, half after half, with an X in between so that the halves do not merge."""
    parts = []
    for p in np.unique(match['period'][match['period'] > 0]):
        parts.append(labels[match['period'] == p])
        parts.append(np.array(['X']))
    return np.concatenate(parts[:-1])


def spell_f1(match, truth, prediction):
    pm = scorer()
    result = pm.evaluate(pm.label_codes(game_sequence(match, truth)),
                         pm.label_codes(game_sequence(match, prediction)), bootstrap_n=0)
    return result['spell_f1_mean'], result['macro_f1']


# --------------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------------

def statistics(match):
    """The same statistics for any match in the common format, as an ordered dictionary."""
    fps, length, width = match['fps'], match['length'], match['width']
    game = match['period'] > 0
    minutes = game.sum() / fps / 60
    seen = np.isfinite(match['bx']) & np.isfinite(match['by']) & game
    tracked = np.isfinite(match['px']) & np.isfinite(match['py'])
    s = OrderedDict()
    s['minutes of play'] = minutes

    # Ball
    s['ball visible (% of frames)'] = 100 * seen[game].mean()
    _, gap_lengths = runs_in_periods(match, ~seen)
    gap_seconds = gap_lengths / fps
    s['ball gaps per 90 minutes'] = len(gap_lengths) / minutes * 90
    s['median gap duration (s)'] = np.median(gap_seconds) if len(gap_seconds) else np.nan
    s['gaps under 5 s (% of gaps)'] = 100 * np.mean(gap_seconds < 5) if len(gap_seconds) else np.nan
    missing = gap_lengths.sum()
    s['frames missing in gaps over 30 s (%)'] = 100 * gap_lengths[gap_seconds > 30].sum() / missing if missing else np.nan
    outside = seen & ((np.abs(match['bx']) > length / 2 + OUT_MARGIN) | (np.abs(match['by']) > width / 2 + OUT_MARGIN))
    _, out_lengths = runs_in_periods(match, outside)
    s['ball out of the pitch, episodes per 10 minutes'] = np.sum(out_lengths >= 5) / minutes * 10
    v_ball = ball_speed(match)
    s['median ball speed (m/s)'] = np.nanmedian(v_ball)
    s['ball above 40 m/s (% of frames)'] = 100 * np.mean(v_ball[np.isfinite(v_ball)] > 40)

    # Players
    count = tracked.sum(1)
    per_team = np.stack([tracked[:, match['team'] == t].sum(1) for t in (HOME, AWAY)], axis=1)
    s['player slots in the file'] = tracked.shape[1]
    s['players tracked per frame (median)'] = np.median(count[game])
    s['frames with 21-22 players tracked (%)'] = 100 * np.mean(count[game] >= 21)
    s['frames with at least 10 players per team (%)'] = 100 * np.mean(per_team[game].min(1) >= 10)
    track_seconds = np.concatenate([runs_in_periods(match, tracked[:, k])[1] for k in range(tracked.shape[1])]) / fps
    s['median track duration (s)'] = np.median(track_seconds) if len(track_seconds) else np.nan
    s['tracks per 90 minutes'] = len(track_seconds) / minutes * 90
    v = player_speeds(match)
    mean_speed = np.nanmean(np.where(tracked, v, np.nan), axis=1)
    inside = seen & ~outside
    s['mean player speed (m/s)'] = np.nanmean(mean_speed[game])
    s['  ... with the ball visible on the pitch'] = np.nanmean(mean_speed[inside])
    s['  ... with the ball visible outside'] = np.nanmean(mean_speed[outside])
    s['  ... with the ball missing'] = np.nanmean(mean_speed[game & ~seen])

    # Player closest to the ball and baseline
    team, d_min = nearest(match)
    has = team >= 0
    s['closest within 2 m (% of frames with the ball visible)'] = 100 * np.mean(d_min[has] < 2)
    pair = has[:-1] & has[1:] & (match['period'][:-1] == match['period'][1:])
    flips = np.sum(team[:-1][pair] != team[1:][pair])
    s['changes of the closest team per minute'] = flips / (pair.sum() / fps / 60) if pair.any() else np.nan
    predictions = {}
    for name, seconds in BASELINE_WINDOWS:
        predictions[name] = baseline(match, team, seconds)
        s['baseline {}: possessions per 10 minutes'.format(name)] = len(possessions(match, predictions[name])) / minutes * 10

    # Only where there is a ground truth
    labels = match.get('labels')
    if labels is not None:
        alive = np.isin(labels, ('H', 'A')) & game
        dead = (labels == 'D') & game
        durations = possessions(match, labels)
        s['live ball (% of frames)'] = 100 * alive[game].mean()
        s['true possessions per 10 minutes'] = len(durations) / minutes * 10
        s['mean duration of a true possession (s)'] = durations.mean()
        s['true possessions under 5 s (%)'] = 100 * np.mean(durations < 5)
        s['ball visible during live play (%)'] = 100 * seen[alive].mean()
        s['ball visible during dead ball (%)'] = 100 * seen[dead].mean()
        s['mean player speed during live play (m/s)'] = np.nanmean(mean_speed[alive])
        s['mean player speed during dead ball (m/s)'] = np.nanmean(mean_speed[dead])
        for name, _ in BASELINE_WINDOWS:
            s['baseline {}: spellF1'.format(name)] = spell_f1(match, labels, predictions[name])[0]
    return s


def compare(*matches):
    """Text table with one column per match and one row per statistic."""
    stats = [statistics(m) for m in matches]
    keys = list(OrderedDict.fromkeys(k for st in stats for k in st))
    width = max(len(k) for k in keys) + 2
    lines = [''.ljust(width) + ''.join('{:>18}'.format(m['name'][:17]) for m in matches)]
    for k in keys:
        cells = []
        for st in stats:
            value = st.get(k)
            if value is None:
                cells.append('{:>18}'.format('–'))
            elif isinstance(value, (int, np.integer)):
                cells.append('{:>18d}'.format(int(value)))
            elif np.isnan(value):
                cells.append('{:>18}'.format('n/a'))
            else:
                cells.append('{:>18.{}f}'.format(value, 3 if 'spellF1' in k else 1))
        lines.append(k.ljust(width) + ''.join(cells))
    return '\n'.join(lines), stats
