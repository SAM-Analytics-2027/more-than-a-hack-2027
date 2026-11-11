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

import numpy as np

# La cartella data/ della repo (quella con tactical/ e altre_partite/); si può cambiare con MTAH_DATA_DIR
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.environ.get('MTAH_DATA_DIR') or os.path.join(ROOT, 'data')
FPS = 25
HOME, AWAY = 0, 1

# Barça–Athletic: costanti della repo ufficiale (games.py e DATA.md)
OUR_FRAMES = 199319
OUR_LENGTH, OUR_WIDTH = 105.3, 68.0
OUR_HALVES = ((1, 11185, 90651), (2, 110808, 190197))  # frame valutati, estremi inclusi


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
            row = int(frame_id) - 1  # frame_id parte da 1, i frame video da 0
            values = np.fromstring(players.replace(';', ','), sep=',', dtype=np.float32).reshape(n_slots, 3)
            px[row], py[row] = values[:, 0], values[:, 1]  # la terza colonna è la velocità: si ricalcola
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
