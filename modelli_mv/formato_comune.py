"""Lettura delle partite nel formato dei file della gara (FIFA EPTS di Metrica).

È la parte di data/analisi/formato_comune.py che serve ai modelli, copiata qui perché data/ non va su GitHub.

Una partita è un dizionario, sempre con gli stessi campi:
    name           nome breve della partita
    home, away     nomi delle due squadre
    fps            frame al secondo (25)
    length, width  dimensioni del campo in metri
    period         per ogni frame: 1 o 2 se è un frame di gioco, 0 altrimenti
    px, py         posizioni dei giocatori in metri, forma (frame, slot); NaN quando lo slot non è tracciato
    team           per ogni slot: 0 = casa, 1 = trasferta
    bx, by         posizione della palla in metri; NaN quando non si vede
    labels         'H', 'A', 'D' o 'X' per ogni frame (solo partite esterne; None per la nostra)

Su disco:
    <nome>_atd.txt              una riga per frame: numero:x,y,velocità;...;x,y,velocità:x,y,velocità palla
    <nome>_atd.xml              squadre, slot dei giocatori, campo, fps, tempi
    <nome>_possession_gt.csv    frame,label (solo partite esterne)

    read_ours()            Barça–Athletic (data/tactical/)
    read_epts(cartella)    una partita in quel formato, per esempio data/altre_partite/dfl_sporcate/J03WMX
"""
import glob
import os
import xml.etree.ElementTree as ET
import csv

import numpy as np

# La cartella data/ della repo (quella con tactical/ e altre_partite/); si può cambiare con MTAH_DATA_DIR
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.environ.get('MTAH_DATA_DIR') or os.path.join(ROOT, 'data')

from constants import FPS, HOME, AWAY, OUR_FRAMES, OUR_LENGTH, OUR_WIDTH, OUR_HALVES


def _only(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError('cercavo {} in {}: trovati {} file'.format(pattern, folder, len(hits)))
    return hits[0]


def read_ours(data_dir=DATA_DIR):
    """Barça–Athletic dai file della gara (tactical/). Campo e tempi dalle costanti della repo ufficiale."""
    return read_epts(os.path.join(data_dir, 'tactical'), name='Barça–Athletic',
                     length=OUR_LENGTH, width=OUR_WIDTH, halves=OUR_HALVES)


def _xml_parts(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.split('}')[-1]
    elements = [(el.tag.split('}')[-1], el) for el in root.iter()]
    params = {}
    for tag, el in elements:
        if tag == 'ProviderParameter':
            params.setdefault(el.findtext('Name'), el.findtext('Value'))
    return elements, params


def read_epts(folder, name=None, length=None, width=None, halves=None, labels_folder=None):
    """Una partita nel formato dei file della gara (FIFA EPTS di Metrica): *_atd.txt e *_atd.xml.

    Se c'è *_possession_gt.csv (frame,label), legge anche le etichette: lo cerca in `labels_folder`, se dato,
    altrimenti nella cartella stessa. Le partite sporcate non hanno il CSV: le loro etichette sono quelle
    della versione pulita, per esempio read_epts('.../dfl_sporcate/J03WMX', labels_folder='.../dfl_pulite/J03WMX').
    I tempi si prendono da `halves` ((tempo, primo frame, ultimo frame), ...) se dato, altrimenti dai
    parametri first/second_half_start/end dell'XML (numeri di frame del file .txt, che partono da 1).
    """
    xml_path = _only(folder, '*_atd.xml')
    elements, params = _xml_parts(xml_path)
    teams = [el for tag, el in elements if tag == 'Team']
    if len(teams) != 2 or not all(el.get('id') for el in teams) or teams[0].get('id') == teams[1].get('id'):
        raise ValueError('XML: servono due squadre distinte con id valido')
    team_ids = {teams[0].get('id'): HOME, teams[1].get('id'): AWAY}
    players_xml = [el for tag, el in elements if tag == 'Player']
    if not players_xml or any(el.get('teamId') not in team_ids for el in players_xml):
        raise ValueError('XML: giocatore senza una squadra riconosciuta')
    team = np.array([team_ids[el.get('teamId')] for el in players_xml], dtype=np.int8)
    n_slots = len(team)
    spec = next(el for tag, el in elements if tag == 'DataFormatSpecification')
    n_frames = int(spec.get('endFrame'))
    if n_frames < 1:
        raise ValueError('XML: numero di frame non positivo')
    rate = next((el.text for tag, el in elements if tag == 'FrameRate'), None)
    if rate is not None and not np.isclose(float(rate), FPS):
        raise ValueError('XML: attesi {} fps; ricampionare esplicitamente il tracking'.format(FPS))
    field = next(el for tag, el in elements if tag == 'FieldSize')
    length = float(field.findtext('Width')) if length is None else float(length)
    width = float(field.findtext('Height')) if width is None else float(width)
    if not (np.isfinite(length) and np.isfinite(width) and length > 0 and width > 0):
        raise ValueError('dimensioni del campo non valide')

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    found = np.zeros(n_frames, dtype=bool)
    with open(_only(folder, '*_atd.txt'), encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError('ATD: formato non valido alla riga {}'.format(line_number))
            frame_id, players, ball = parts
            row = int(frame_id) - 1  # frame_id parte da 1, i frame video da 0
            if row < 0 or row >= n_frames or found[row]:
                raise ValueError('ATD: frame duplicato o fuori intervallo alla riga {}'.format(line_number))
            found[row] = True
            values = np.fromstring(players.replace(';', ','), sep=',', dtype=np.float32)
            if values.size != n_slots * 3:
                raise ValueError('ATD: numero di coordinate giocatori errato alla riga {}'.format(line_number))
            values = values.reshape(n_slots, 3)
            px[row], py[row] = values[:, 0], values[:, 1]  # la terza colonna è la velocità: si ricalcola
            ball_values = np.fromstring(ball, sep=',', dtype=np.float32)
            if ball_values.size != 3:
                raise ValueError('ATD: coordinate palla non valide alla riga {}'.format(line_number))
            bx[row], by[row] = ball_values[0], ball_values[1]
    # Missing ATD rows remain explicitly unobserved. A coordinate pair is usable
    # only when both axes are finite (infinity is not a physical position).
    missing_players = ~np.isfinite(px) | ~np.isfinite(py)
    px[missing_players] = np.nan
    py[missing_players] = np.nan
    missing_ball = ~np.isfinite(bx) | ~np.isfinite(by)
    bx[missing_ball] = np.nan
    by[missing_ball] = np.nan

    period = np.zeros(n_frames, dtype=np.int8)
    if halves is None:
        halves = [(half, int(params[key + '_start']) - 1, int(params[key + '_end']) - 1)
                  for half, key in ((1, 'first_half'), (2, 'second_half')) if params.get(key + '_start')]
    for half, first, last in halves:
        if half not in (1, 2) or first < 0 or last < first or last >= n_frames:
            raise ValueError('XML: intervallo di tempo non valido')
        if period[first:last + 1].any():
            raise ValueError('XML: tempi di gioco sovrapposti')
        period[first:last + 1] = half

    labels = None
    gt = glob.glob(os.path.join(labels_folder or folder, '*_possession_gt.csv'))
    if len(gt) > 1:
        raise ValueError('più file ground truth nella stessa cartella')
    if labels_folder is not None and not gt:
        raise FileNotFoundError('ground truth mancante in {}'.format(labels_folder))
    if gt:
        labels = np.full(n_frames, 'X', dtype='<U1')
        labelled = np.zeros(n_frames, dtype=bool)
        with open(gt[0], newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['frame', 'label']:
                raise ValueError('ground truth: intestazione attesa frame,label')
            for row in reader:
                frame = int(row['frame'])
                letter = row['label']
                if frame < 0 or frame >= n_frames or labelled[frame] or letter not in ('H', 'A', 'D', 'X'):
                    raise ValueError('ground truth: frame duplicato/fuori intervallo o etichetta non valida')
                labels[frame] = letter
                labelled[frame] = True
    competition = next((el for tag, el in elements if tag == 'Competition'), None)
    info = {  # informazioni della partita, per riscriverla uguale con save_epts
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
