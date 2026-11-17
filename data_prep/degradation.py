"""Degradation: giving the external matches the same defects as the tracking of the competition match.

It is done one piece at a time (each piece was measured, tuned and checked against the competition tracking):
    1. missing ball                 remove_ball
    2. shifted ball                 shift_ball
    3. missing players and slots    remove_players

Rules:
- the parameters are tuned only by comparing statistics of the competition tracking, never its labels;
- the labels do not change: degrading only changes what the model sees;
- it is reproducible: for every match and every piece the random generator starts from a fixed seed.
"""
import itertools
import zlib

import numpy as np

import match_format as fc

SEED = 2027

# Seed tag of each piece. The published degraded data were produced with these exact tags, so they must not
# change: a different tag gives a different random draw and different files.
PIECE_SEED_TAGS = {'ball': 'palla', 'shift': 'spostamento', 'players': 'giocatori'}

# Bands of mean player speed (m/s): they say "when" the ball goes missing
SPEED_BANDS = (0, 1.0, 1.3, 1.6, 2.0, 99)

# Piece 1, missing ball: the recipe, found with calibrate_ball
BALL = {
    'pause_share': 0.8,       # share of pauses in which the ball disappears
    'delay': 0.0,             # s: on average the ball disappears this long after the pause starts (negative = before)
    'early': 1.0,             # s: on average it reappears this long before the restart (negative = after)
                              # (delay and early vary at random by ±1.5 s from one pause to the next)
    'live_gaps': 1.5,         # gaps during live play per minute of live play
    'live_duration': 4.0,     # s: median duration of a gap during live play
    'live_variability': 0.5,  # how much the durations of the live gaps vary (0.5 = a little, 1 = a lot)
}
# Grid with which it was found (calibrate_ball on the 7 DFL matches, 864 combinations)
BALL_GRID = {'pause_share': [0.7, 0.8, 0.9], 'delay': [-1, 0, 1], 'early': [-1, 0, 1],
             'live_gaps': [1.0, 1.5, 2.0, 3.0], 'live_duration': [2, 3, 4, 6], 'live_variability': [0.5, 1.0]}
ZONE_WEIGHT = {'penalty box': 3.0, 'within 5 m of the touchline': 3.0, 'rest of the pitch': 1.0, 'outside the pitch': 1.0}
ZONES = ('outside the pitch', 'penalty box', 'within 5 m of the touchline', 'rest of the pitch')


def seed_for(match, piece):
    """Fixed random generator for (match, piece): same seed, same result."""
    return np.random.default_rng([SEED, zlib.crc32(match['name'].encode()),
                                  zlib.crc32(PIECE_SEED_TAGS[piece].encode())])


def zone(match, x, y):
    """Index in ZONES for each position (metres, origin at the centre)."""
    L, W = match['length'], match['width']
    outside = (np.abs(x) > L / 2) | (np.abs(y) > W / 2)
    box = (np.abs(x) > L / 2 - 16.5) & (np.abs(y) < 20.16)
    side = np.abs(y) > W / 2 - 5
    return np.where(outside, 0, np.where(box, 1, np.where(side, 2, 3)))


def mean_speed(match):
    """Mean speed of the tracked players, frame by frame (m/s)."""
    return np.nanmean(np.where(np.isfinite(match['px']), fc.player_speeds(match), np.nan), axis=1)


# --------------------------------------------------------------------------------------------------
# Piece 1: missing ball
# --------------------------------------------------------------------------------------------------

def remove_ball(match, p=None, rng=None):
    """Copy of the match with the ball removed as in the competition tracking: gaps in the pauses and short gaps
    during live play.

    The pauses come from the labels of the external match (D). The labels stay the same.
    """
    p = dict(BALL, **(p or {}))
    rng = rng or seed_for(match, 'ball')
    fps, labels, period = match['fps'], match['labels'], match['period']
    n = len(labels)
    missing = np.zeros(n, dtype=bool)
    start_of = {q: np.flatnonzero(period == q)[0] for q in np.unique(period[period > 0])}     # start of each half
    end_of = {q: np.flatnonzero(period == q)[-1] + 1 for q in np.unique(period[period > 0])}  # end of each half

    # 1. Gaps in the pauses: the ball disappears a few seconds after the stop and reappears around the restart
    starts, lengths = fc.runs_in_periods(match, labels == 'D')
    for start, length in zip(starts, lengths):
        if rng.random() >= p['pause_share']:
            continue
        delay = rng.uniform(p['delay'] - 1.5, p['delay'] + 1.5)
        early = rng.uniform(p['early'] - 1.5, p['early'] + 1.5)
        a = max(start + int(round(delay * fps)), start_of[period[start]])
        b = min(start + length - int(round(early * fps)), end_of[period[start]])
        if b - a >= fps // 5:  # at least 0.2 s
            missing[a:b] = True

    # 2. Gaps during live play: more likely where the ball is in the box or close to the lines
    alive = np.isin(labels, ('H', 'A')) & (period > 0)
    weight = np.array([ZONE_WEIGHT[z] for z in ZONES])[zone(match, match['bx'], match['by'])]
    weight = np.where(alive & np.isfinite(match['bx']), weight, 0.0)
    count = rng.poisson(p['live_gaps'] * alive.sum() / fps / 60)
    if count and weight.sum() > 0:
        begins = rng.choice(n, size=count, replace=False, p=weight / weight.sum())
        for begin, seconds in zip(begins, rng.lognormal(np.log(p['live_duration']), p['live_variability'], size=count)):
            missing[begin:min(begin + max(1, int(round(seconds * fps))), end_of[period[begin]])] = True

    out = dict(match)
    out['bx'] = np.where(missing, np.nan, match['bx']).astype(np.float32)
    out['by'] = np.where(missing, np.nan, match['by']).astype(np.float32)
    return out


def ball_fingerprint(match, speed=None, ignore_flicker=0):
    """The measures of the missing ball that the degradation must copy (piece 1).

    ignore_flicker=N: gaps of at most N frames (the ball "flickering") do not count as gaps but as a visible
    ball. It is meant for sources such as Metrica, which have many more of them than the competition tracking:
    this way only the real gaps are compared. With 0 (the default) all of them count, as for DFL.
    """
    speed = mean_speed(match) if speed is None else speed
    fps = match['fps']
    game = match['period'] > 0
    seen = np.isfinite(match['bx']) & game
    if ignore_flicker:
        flicker_starts, flicker_lengths = fc.runs_in_periods(match, ~seen)
        for start, length in zip(flicker_starts, flicker_lengths):
            if length <= ignore_flicker:
                seen[start:start + length] = True
    minutes = game.sum() / fps / 60
    out = {'ball missing (%)': 100 * np.mean(~seen[game])}
    for lo, hi in zip(SPEED_BANDS[:-1], SPEED_BANDS[1:]):
        sel = game & (speed >= lo) & (speed < hi)
        name = 'players above {:.1f} m/s'.format(lo) if hi > 50 else 'players {:.1f}-{:.1f} m/s'.format(lo, hi)
        out['  missing with ' + name + ' (%)'] = 100 * np.mean(~seen[sel])
    gap_starts, gaps = fc.runs_in_periods(match, ~seen)
    _, visible = fc.runs_in_periods(match, seen)
    out['gaps per 90 minutes'] = len(gaps) / minutes * 90
    for q in (25, 50, 75):
        out['gap duration, p{} (s)'.format(q)] = np.percentile(gaps / fps, q) if len(gaps) else np.nan
    for q in (25, 50, 75):
        out['visible stretches, p{} (s)'.format(q)] = np.percentile(visible / fps, q) if len(visible) else np.nan
    last = gap_starts - 1
    last = last[(last >= 0) & np.isfinite(match['bx'][np.maximum(last, 0)])]
    z = zone(match, match['bx'][last], match['by'][last])
    for k, name in enumerate(ZONES):
        out['gap starting: {} (%)'.format(name)] = 100 * np.mean(z == k) if len(z) else np.nan
    return out


# Measures used to tune the parameters, with their weight
BALL_TARGETS = {
    'ball missing (%)': 3.0,
    '  missing with players 0.0-1.0 m/s (%)': 1.0, '  missing with players 1.0-1.3 m/s (%)': 1.0,
    '  missing with players 1.3-1.6 m/s (%)': 1.0, '  missing with players 1.6-2.0 m/s (%)': 1.0,
    '  missing with players above 2.0 m/s (%)': 1.0,
    'gaps per 90 minutes': 2.0,
    'gap duration, p25 (s)': 1.0, 'gap duration, p50 (s)': 1.0, 'gap duration, p75 (s)': 1.0,
    'visible stretches, p25 (s)': 1.0, 'visible stretches, p50 (s)': 1.0, 'visible stretches, p75 (s)': 1.0,
    'gap starting: outside the pitch (%)': 1.0,
}


def distance(measures, target, weights=BALL_TARGETS):
    """How far the measures are from the target: weighted sum of the squared relative errors."""
    return sum(w * ((measures[k] - target[k]) / target[k]) ** 2 for k, w in weights.items())


def mean_of(items):
    """Mean, measure by measure, of a list of fingerprints (one per match)."""
    return {k: float(np.nanmean([m[k] for m in items])) for k in items[0]}


def calibrate(degrade, fingerprint, weights, matches, target, grid, speeds=None):
    """Tries every combination of the grid on the matches and returns the best ones, in order.

    degrade(match, parameters) degrades a match; fingerprint(match, speed) measures it. Each combination
    is judged with distance() between the mean fingerprint of the degraded matches and the target (the
    competition tracking).
    """
    speeds = speeds or [mean_speed(m) for m in matches]
    results = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid, combo))
        mean = mean_of([fingerprint(degrade(m, p), s) for m, s in zip(matches, speeds)])
        results.append((distance(mean, target, weights), p, mean))
    return sorted(results, key=lambda r: r[0])


def calibrate_ball(matches, target, grid, speeds=None):
    """Tuning of piece 1 (missing ball)."""
    return calibrate(remove_ball, ball_fingerprint, BALL_TARGETS, matches, target, grid, speeds)


# --------------------------------------------------------------------------------------------------
# Piece 2: shifted ball
# --------------------------------------------------------------------------------------------------

# The recipe, found with calibrate_shift
SHIFT = {
    'amplitude': 0.8,         # m: how far the ball is shifted (standard deviation on each axis)
    'timescale': 0.5,         # s: how slowly the shift changes (higher = slower)
    'delay': 2,               # frames: how late the ball arrives with respect to the players (2 = 0.08 s)
}
# Grid with which it was found (calibrate_shift on the 7 DFL matches already degraded by piece 1)
SHIFT_GRID = {'amplitude': [0.6, 0.7, 0.8, 0.9], 'timescale': [0.25, 0.35, 0.5, 0.75], 'delay': [0, 1, 2]}


def shift_ball(match, p=None, rng=None):
    """Copy of the match with the ball shifted by a slowly changing error, plus an optional delay.

    The shift is a "smoothed" random noise (Gaussian filter), independent on the two axes: its standard
    deviation is `amplitude` metres and it changes over about `timescale` seconds. Being smooth, it moves
    the ball without making it shake from one frame to the next.
    """
    from scipy.ndimage import gaussian_filter1d

    p = dict(SHIFT, **(p or {}))
    rng = rng or seed_for(match, 'shift')
    fps, period = match['fps'], match['period']
    n = len(period)
    offsets = []
    for _ in range(2):
        smooth = gaussian_filter1d(rng.standard_normal(n), sigma=p['timescale'] * fps, mode='wrap')
        offsets.append(smooth / smooth.std() * p['amplitude'])
    bx, by = match['bx'].astype(np.float64), match['by'].astype(np.float64)
    lag = int(p['delay'])
    if lag:
        # The ball at frame t is the true one of lag frames earlier, only within the same half
        shifted_x, shifted_y = np.full(n, np.nan), np.full(n, np.nan)
        shifted_x[lag:], shifted_y[lag:] = bx[:-lag], by[:-lag]
        same = np.zeros(n, dtype=bool)
        same[lag:] = period[lag:] == period[:-lag]
        bx, by = np.where(same, shifted_x, np.nan), np.where(same, shifted_y, np.nan)
    out = dict(match)
    out['bx'] = (bx + offsets[0]).astype(np.float32)
    out['by'] = (by + offsets[1]).astype(np.float32)
    return out


def clear_live_play(match, speed):
    """Frames of clearly live play: ball visible on the pitch and players above 1.6 m/s (and live per the labels)."""
    game = match['period'] > 0
    seen = np.isfinite(match['bx']) & game
    inside = seen & (np.abs(match['bx']) <= match['length'] / 2) & (np.abs(match['by']) <= match['width'] / 2)
    live = inside & (speed > 1.6)
    if match['labels'] is not None:
        live &= np.isin(match['labels'], ('H', 'A'))
    return live


def shift_fingerprint(match, speed=None):
    """The measures of the shifted ball, during clearly live play (piece 2)."""
    speed = mean_speed(match) if speed is None else speed
    live = clear_live_play(match, speed)
    team, d = fc.nearest(match)
    v = fc.ball_speed(match, 2)
    out = {'distance from the closest player, median (m)': np.median(d[live]),
           'closest within 1 m (%)': 100 * np.mean(d[live] < 1),
           'closest within 2 m (%)': 100 * np.mean(d[live] < 2)}
    for name, sel in (('slow ball, < 3 m/s', v < 3), ('medium ball, 3-10 m/s', (v >= 3) & (v < 10)), ('fast ball, > 10 m/s', v >= 10)):
        out['  median distance, {} (m)'.format(name)] = np.median(d[live & sel])
    kernel = np.ones(9) / 9
    mx = np.convolve(np.nan_to_num(match['bx']), kernel, mode='same')
    my = np.convolve(np.nan_to_num(match['by']), kernel, mode='same')
    full = np.convolve(np.isfinite(match['bx']).astype(float), kernel, mode='same') > 0.999
    out['ball jitter (cm)'] = 100 * np.median(np.hypot(match['bx'] - mx, match['by'] - my)[live & full])
    out['median ball speed (m/s)'] = np.nanmedian(fc.ball_speed(match)[live])
    pair = live[:-1] & live[1:] & (team[:-1] >= 0) & (team[1:] >= 0)
    out['changes of the closest team per minute'] = np.sum(team[:-1][pair] != team[1:][pair]) / (pair.sum() / match['fps'] / 60)
    return out


SHIFT_TARGETS = {
    'distance from the closest player, median (m)': 2.0, 'closest within 1 m (%)': 1.0, 'closest within 2 m (%)': 1.0,
    '  median distance, slow ball, < 3 m/s (m)': 1.0, '  median distance, medium ball, 3-10 m/s (m)': 1.0,
    '  median distance, fast ball, > 10 m/s (m)': 1.0,
    'ball jitter (cm)': 1.0, 'median ball speed (m/s)': 1.0,
}


def calibrate_shift(matches, target, grid, speeds=None):
    """Tuning of piece 2 (shifted ball), on the matches already degraded by piece 1."""
    return calibrate(shift_ball, shift_fingerprint, SHIFT_TARGETS, matches, target, grid, speeds)


# --------------------------------------------------------------------------------------------------
# Piece 3: players who disappear and anonymous tracks
# --------------------------------------------------------------------------------------------------

CLOSE = 1.5         # m: a player is "close" to another below this distance
SLOT = (24, 22)     # slots per team in the degraded file, as in the competition match (24 home, 22 away)

# How often a player is lost, per minute of track, in the four situations (close to another
# player?, ball missing?). Measured on the competition tracking with loss_rates.
LOSS_RATES = {(False, False): 0.189,   # far from the others, ball visible
              (False, True): 0.486,    # far from the others, ball missing
              (True, False): 2.266,    # close to another, ball visible
              (True, True): 4.463}     # close to another, ball missing

# The recipe: the remaining numbers, found with calibrate_players
PLAYERS = {
    'scale': 1.4,                   # multiplies the measured losses (1 = as measured)
    'hidden': 2.0,                  # s: how long a lost player stays invisible, median
    'hidden_variability': 0.8,      # how much this time varies from one loss to the next
}
# Grid with which it was found (calibrate_players on the 7 DFL matches already degraded by pieces 1 and 2).
# Open point: with this recipe the tracks last almost as long as in the competition tracking, but there are more
# losses (about 1,500 against 1,000 per 90 minutes); with fewer losses the tracks become too long. Neither "ghost"
# tracks nor a longer invisibility during dead ball improve the similarity.
PLAYERS_GRID = {'scale': [1.0, 1.2, 1.4, 1.6, 2.0], 'hidden': [1.5, 2, 3], 'hidden_variability': [0.8, 1.2]}


def runs_by_column(mask, period):
    """Continuous True stretches of each column, within the same half: (column, first frame, last frame)."""
    same = np.zeros(len(period), dtype=bool)
    same[1:] = (period[1:] == period[:-1]) & (period[1:] > 0)
    before = np.zeros_like(mask)
    before[1:] = mask[:-1] & same[1:, None]
    after = np.zeros_like(mask)
    after[:-1] = mask[1:] & same[1:, None]
    col, start = np.nonzero((mask & ~before).T)
    _, end = np.nonzero((mask & ~after).T)
    return col, start, end


def close_to_other(match):
    """For each frame and column: True if another player (of either team) is within CLOSE metres."""
    px, py = match['px'], match['py']
    out = np.zeros(px.shape, dtype=bool)
    for j in range(px.shape[1]):
        d = np.hypot(px - px[:, [j]], py - py[:, [j]])
        d[:, j] = np.inf
        out[:, j] = (np.where(np.isfinite(d), d, np.inf) < CLOSE).any(1)
    return out


def loss_rates(match, near=None):
    """Losses per minute of track in the four situations (close to another player?, ball missing?)."""
    near = close_to_other(match) if near is None else near
    period = match['period']
    game = period > 0
    tracked = np.isfinite(match['px']) & game[:, None]
    ball_missing = ~np.isfinite(match['bx'])
    col, start, end = runs_by_column(tracked, period)
    nxt = np.minimum(end + 1, len(period) - 1)
    lost = (end + 1 < len(period)) & (period[nxt] == period[end])  # does not end with the half
    out = {}
    for v in (False, True):
        for m in (False, True):
            minutes = (tracked & (near == v) & (ball_missing == m)[:, None]).sum() / match['fps'] / 60
            events = np.sum(lost & (near[end, col] == v) & (ball_missing[end] == m))
            out[(v, m)] = events / minutes if minutes else 0.0
    return out


def remove_players(match, p=None, rng=None, near=None, rates=None):
    """Copy of the match with lost players and anonymous tracks, as in the competition tracking.

    Each player is lost with the frequency measured there in his situation (close to another, ball
    missing); he stays invisible for a few seconds and reappears as a new track, in a random free slot of
    his team. The degraded file has SLOT[0] + SLOT[1] columns, like the competition file.
    """
    p = dict(PLAYERS, **(p or {}))
    rates = rates or LOSS_RATES
    rng = rng or seed_for(match, 'players')
    fps, period, team = match['fps'], match['period'], match['team']
    n, k = match['px'].shape
    near = close_to_other(match) if near is None else near
    ball_missing = ~np.isfinite(match['bx'])

    rate = np.zeros((n, k))
    for (v, m), r in rates.items():
        rate[(near == v) & (ball_missing == m)[:, None]] = r
    hazard = rate * p['scale'] / (60 * fps)          # probability of losing him in each frame
    on = np.isfinite(match['px']) & (period > 0)[:, None]

    # 1. The visible stretches of each player
    segments = []  # (team, first frame, last frame, column)
    col, start, end = runs_by_column(on, period)
    for j in range(k):
        breaks = np.flatnonzero(on[:, j] & (rng.random(n) < hazard[:, j]))
        for a, b in zip(start[col == j], end[col == j]):
            t = a
            while t <= b:
                i = np.searchsorted(breaks, t + 1)
                cut = breaks[i] if i < len(breaks) and breaks[i] <= b else None
                if cut is None:
                    segments.append((team[j], t, b, j))
                    break
                segments.append((team[j], t, cut, j))
                hidden = rng.lognormal(np.log(p['hidden']), p['hidden_variability'])
                t = cut + 1 + max(1, int(round(hidden * fps)))

    # 2. Each stretch in a random free slot of its team (at least one empty frame between two tracks)
    first = {fc.HOME: 0, fc.AWAY: SLOT[0]}
    free_from = {fc.HOME: np.zeros(SLOT[0], dtype=int), fc.AWAY: np.zeros(SLOT[1], dtype=int)}
    px = np.full((n, sum(SLOT)), np.nan, dtype=np.float32)
    py = np.full((n, sum(SLOT)), np.nan, dtype=np.float32)
    for t_team, a, b, j in sorted(segments, key=lambda s: (s[1], s[3])):
        free = np.flatnonzero(free_from[t_team] <= a)
        if not len(free):
            raise ValueError('too many players at the same time for the slots of the team')
        s = rng.choice(free)
        free_from[t_team][s] = b + 2
        px[a:b + 1, first[t_team] + s] = match['px'][a:b + 1, j]
        py[a:b + 1, first[t_team] + s] = match['py'][a:b + 1, j]

    out = dict(match)
    out['px'], out['py'] = px, py
    out['team'] = np.array([fc.HOME] * SLOT[0] + [fc.AWAY] * SLOT[1], dtype=np.int8)
    return out


def players_fingerprint(match, speed=None):
    """The measures of the players and of the tracks (piece 3)."""
    fps, period, team = match['fps'], match['period'], match['team']
    n = len(period)
    game = period > 0
    minutes = game.sum() / fps / 60
    tracked = np.isfinite(match['px']) & game[:, None]
    seen = np.isfinite(match['bx']) & game
    count = tracked.sum(1)
    out = {'22 or more players tracked (%)': 100 * np.mean(count[game] >= 22),
           '21 players tracked (%)': 100 * np.mean(count[game] == 21),
           '20 or fewer players tracked (%)': 100 * np.mean(count[game] <= 20),
           'fewer than 21, with the ball visible (%)': 100 * np.mean(count[seen] < 21),
           'fewer than 21, with the ball missing (%)': 100 * np.mean(count[game & ~seen] < 21) if (game & ~seen).any() else np.nan}
    col, start, end = runs_by_column(tracked, period)
    lengths = (end - start + 1) / fps
    out['tracks per 90 minutes'] = len(col) / minutes * 90
    for q in (25, 50, 75):
        out['track duration, p{} (s)'.format(q)] = np.percentile(lengths, q)
    nxt = np.minimum(end + 1, n - 1)
    lost = (end + 1 < n) & (period[nxt] == period[end])
    le, lc = end[lost], col[lost]
    out['tracks lost per 90 minutes'] = lost.sum() / minutes * 90
    x, y = match['px'][le, lc], match['py'][le, lc]
    d = np.hypot(match['px'][le] - x[:, None], match['py'][le] - y[:, None])
    d[np.arange(len(le)), lc] = np.inf
    out['lost close to another player (%)'] = 100 * np.mean(np.where(np.isfinite(d), d, np.inf).min(1) < CLOSE)
    out['lost with the ball missing (%)'] = 100 * np.mean(~seen[le])
    # Reappears: another track of the same team starts within 30 s and within 10 m of where it was lost
    order = np.argsort(start, kind='stable')
    s_time, s_col = start[order], col[order]
    hidden = np.full(len(le), np.nan)
    for i, (e, c) in enumerate(zip(le, lc)):
        lo, hi = np.searchsorted(s_time, e + 1), np.searchsorted(s_time, e + 1 + 30 * fps)
        t, cc = s_time[lo:hi], s_col[lo:hi]
        ok = (team[cc] == team[c]) & (cc != c)
        if ok.any():
            close = np.hypot(match['px'][t[ok], cc[ok]] - x[i], match['py'][t[ok], cc[ok]] - y[i]) < 10
            if close.any():
                hidden[i] = (t[ok][close].min() - e - 1) / fps
    out['reappears within 30 s (%)'] = 100 * np.mean(np.isfinite(hidden))
    found = hidden[np.isfinite(hidden)]
    for q in (25, 50, 75):
        out['time invisible, p{} (s)'.format(q)] = np.percentile(found, q) if len(found) else np.nan
    return out


PLAYERS_TARGETS = {
    '22 or more players tracked (%)': 1.0, '21 players tracked (%)': 1.0, '20 or fewer players tracked (%)': 1.0,
    'fewer than 21, with the ball visible (%)': 1.0, 'fewer than 21, with the ball missing (%)': 1.0,
    'tracks lost per 90 minutes': 2.0,
    'track duration, p25 (s)': 1.0, 'track duration, p50 (s)': 1.0, 'track duration, p75 (s)': 1.0,
    'time invisible, p25 (s)': 1.0, 'time invisible, p50 (s)': 1.0, 'time invisible, p75 (s)': 1.0,
}
# 'reappears within 30 s' is kept only as a check: in the competition tracking 36% of the lost tracks do not
# reappear nearby, but they are probably duplicate or ghost tracks (the frames with 23 players), a defect that
# the recipe does not copy.


def calibrate_players(matches, target, grid, speeds=None, nears=None):
    """Tuning of piece 3, on the matches already degraded by pieces 1 and 2. nears: close_to_other() already computed."""
    nears = nears or {m['name']: close_to_other(m) for m in matches}
    degrade_one = lambda m, p: remove_players(m, p, near=nears[m['name']])
    return calibrate(degrade_one, players_fingerprint, PLAYERS_TARGETS, matches, target, grid, speeds)


# --------------------------------------------------------------------------------------------------
# All together: degrading a match and saving the degraded matches (DFL, or another source)
# --------------------------------------------------------------------------------------------------

PIECES = ('ball', 'shift', 'players')


def degrade(match, pieces=PIECES, recipes=None):
    """The chosen pieces, in order: missing ball, shifted ball, players and slots. Fixed seeds: same match,
    same result.

    pieces:  which ones to apply, for example ('shift', 'players') if in a source the ball is already missing
             as in the competition tracking;
    recipes: numbers different from the ones written above, for example tuned on another source:
             {'ball': {...}, 'shift': {...}, 'players': {...}}.
    """
    recipes = recipes or {}
    if 'ball' in pieces:
        match = remove_ball(match, recipes.get('ball'))
    if 'shift' in pieces:
        match = shift_ball(match, recipes.get('shift'))
    if 'players' in pieces:
        match = remove_players(match, recipes.get('players'))
    return match


def save_degraded(clean_dir, degraded_dir, pieces=PIECES, recipes=None):
    """Reads every clean match in `clean_dir`/<ID>/ and saves the degraded one in `degraded_dir`/<ID>/, in the
    same format as the competition files: only atd.txt and atd.xml, because the labels are the same as the
    clean ones and stay only there. The ball speed stays empty, as almost always in the competition file."""
    import os
    import time

    for match_id in sorted(os.listdir(clean_dir)):
        if not os.path.isdir(os.path.join(clean_dir, match_id)):
            continue
        started = time.time()
        clean = fc.read_epts(os.path.join(clean_dir, match_id))
        dirty = degrade(clean, pieces, recipes)
        info = clean['info']
        fc.save_epts(dict(dirty, labels=None), os.path.join(degraded_dir, match_id), match_id, team_ids=info['team_ids'],
                     home_attacks=info['home_attacks'], session_info=info, ball_speed=False)
        print('degraded {}: ball missing {:.0f}%, {} slots, in {:.0f} s'.format(
            match_id, 100 * np.mean(~np.isfinite(dirty['bx'])), dirty['px'].shape[1], time.time() - started))
    print('ready in', degraded_dir)


if __name__ == '__main__':
    # python data_prep/degradation.py                         the 7 DFL matches: from data/external/dfl_clean to data/external/dfl_degraded
    # python data_prep/degradation.py <clean> <degraded>      another source, with all three pieces and the recipes written above
    import os
    import sys

    base = os.path.join(fc.DATA_DIR, 'external')
    save_degraded(sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, 'dfl_clean'),
                  sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, 'dfl_degraded'))
