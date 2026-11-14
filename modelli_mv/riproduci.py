"""Il modello mv_hgb_rete3 (media di HGB + rete neurale a 3 semi, poi Viterbi), riproducibile su ogni computer.

    python modelli_mv/riproduci.py controlla [--solo-barca]   versioni delle librerie e impronte dei file dei dati
    python modelli_mv/riproduci.py prevedi                    CSV del Barça dai pesi congelati (meno di 1 min)
    python modelli_mv/riproduci.py esterne                    risposte sulle 10 partite DFL + Metrica dai pesi
                                                              congelati, per la seconda pagina della dashboard (5 min)
    python modelli_mv/riproduci.py allena --nome NOME         rifà tutto da capo con il codice di oggi: 5 gruppi
                                                              di validazione, parametri del Viterbi, modello finale
                                                              (circa 1 ora su Mac); i pesi vanno in
                                                              data/cache_mv/modelli/NOME e i risultati in dashboard
                                                              come NOME, accanto al riferimento
    python modelli_mv/riproduci.py prevedi --modello NOME     come prevedi ed esterne, ma con i pesi della prova NOME
    python modelli_mv/riproduci.py esterne --modello NOME     (per rifare identica la prova di un compagno)

Il riferimento sta in modelli_mv/congelato/: pesi dei modelli, parametri del Viterbi e impronte SHA-256 di tutti i
risultati attesi. Ogni prova ha lo stesso, in data/cache_mv/modelli/NOME/. "prevedi" ed "esterne" dicono se quello
che esce è IDENTICO a quello atteso per quei pesi.
Né i CSV né i dati vanno su GitHub: solo codice, pesi e impronte.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dati  # noqa: E402
import esperimento as E  # noqa: E402
import modello as M  # noqa: E402

from constants import (  # noqa: E402
    MODEL_NAME as NAME,
    SEED,
    N_NETS,
    TV_WEIGHT,
    VITERBI_SEARCH_START as START,
    DM_SOURCES as DM,
    LETTERS,
    DECIMALS,
)

ROOT = os.path.normpath(os.path.join(HERE, '..'))
FROZEN = os.path.join(HERE, 'congelato')
RESULTS = os.path.join(ROOT, 'evaluation', 'results')
EXT_RESULTS = os.path.join(dati.C.fc.DATA_DIR, 'risultati_esterne')
PUBLIC_GT = os.path.join(ROOT, 'evaluation', 'gt', 'BAR-ATH_possession_gt_public.csv')


def _sha256(path, text=False):
    """Impronta di un file; per i file di testo gli a capo Windows (\\r\\n) contano come quelli Mac (\\n)."""
    with open(path, 'rb') as f:
        data = f.read()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if text else data).hexdigest()


def _versions():
    out = {'python': platform.python_version(), 'sistema': platform.platform()}
    for pkg in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'numba', 'torch'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_csv(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    M.to_codes(codes)
    tmp = path + '.{}.tmp'.format(os.getpid())
    pd.DataFrame({'frame': np.arange(len(codes)), 'label': LETTERS[codes]}).to_csv(tmp, index=False, lineterminator='\n')
    os.replace(tmp, path)
    return _sha256(path)


# --------------------------------------------------------------------------------------------------
# controlla
# --------------------------------------------------------------------------------------------------

def check(write=False, only_barca=False):
    ok = True
    print('1. Versioni (requirements.txt)')
    if not platform.python_version().startswith('3.11.'):
        print('   ATTENZIONE python {}: serve 3.11'.format(platform.python_version()))
        ok = False
    for line in open(os.path.join(os.path.dirname(HERE), 'requirements.txt'), encoding='utf-8'):
        line = line.split('#')[0].strip()
        if '==' not in line:
            continue
        pkg, want = line.split('==')
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        same = have is not None and have.split('+')[0] == want
        ok &= same
        print('   {:14s} {:10s} {}'.format(pkg, str(have), 'ok' if same else '<-- serve ' + want))

    print('2. File dei dati (impronte in modelli_mv/congelato/dati_attesi.json)')
    manifest_path = os.path.join(FROZEN, 'dati_attesi.json')
    items = [dati.BARCA] + ([] if only_barca else dati.external_matches())
    files = [f for it in items for f in dati.input_files(it)]
    if write:
        manifest = {f: _sha256(os.path.join(dati.C.fc.DATA_DIR, f), text=True) for f in files}
        os.makedirs(FROZEN, exist_ok=True)
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
        print('   scritte le impronte di {} file'.format(len(manifest)))
        return ok
    manifest = json.load(open(manifest_path))
    missing, different = [], []
    for f in files:
        path = os.path.join(dati.C.fc.DATA_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
        elif _sha256(path, text=True) != manifest.get(f):
            different.append(f)
    print('   {} file controllati: {} uguali, {} mancanti, {} diversi'.format(
        len(files), len(files) - len(missing) - len(different), len(missing), len(different)))
    for f in missing[:10]:
        print('   MANCA    data/' + f)
    for f in different[:10]:
        print('   DIVERSO  data/' + f)
    ok &= not missing and not different
    print('Tutto in ordine.' if ok else 'Qualcosa non torna: vedi sopra (e la sezione "Problemi" del README).')
    return ok


# --------------------------------------------------------------------------------------------------
# I quattro pezzi dell'ensemble: allenare, salvare, caricare, prevedere
# --------------------------------------------------------------------------------------------------

def fit_components(data, items, cols, folder, net_seeds):
    """Allena HGB + le reti sulle partite `items` e salva i pesi in `folder`."""
    import joblib
    import famiglie
    import rete
    famiglie.TV_WEIGHT = rete.TV_WEIGHT = TV_WEIGHT  # la ricetta si decide qui, in un posto solo
    os.makedirs(folder, exist_ok=True)
    t = time.time()
    X, y, w = famiglie.rows(data, items, cols)       # 1 frame ogni 8, squadre scambiate, pesi (SkillCorner 1/4)
    hgb = E.make_model().fit(X, y, sample_weight=w)
    joblib.dump(hgb, os.path.join(folder, 'hgb.joblib'), compress=3)
    del X, y, w
    dev = rete.device()
    for s, seed in enumerate(net_seeds):
        rete.save(rete.train(data, items, cols, seed, dev=dev), os.path.join(folder, 'rete_seme{}.pt'.format(s)))
    json.dump({'partite': ['{}/{}'.format(*it) for it in items], 'semi_rete': [int(s) for s in net_seeds],
               'colonne': list(cols), 'dispositivo_allenamento': str(dev), 'feature_version': 2},
              open(os.path.join(folder, 'modello.json'), 'w'), indent=1)
    print('   {}: HGB {} iterazioni + {} reti, {:.0f} s'.format(os.path.relpath(folder, ROOT), hgb.n_iter_,
                                                              len(net_seeds), time.time() - t), flush=True)


def load_components(folder):
    import joblib
    import rete
    info = json.load(open(os.path.join(folder, 'modello.json')))
    hgb = joblib.load(os.path.join(folder, 'hgb.joblib'))
    nets = [rete.load(os.path.join(folder, 'rete_seme{}.pt'.format(s))) for s in range(len(info['semi_rete']))]
    return hgb, nets, info['colonne']


def legacy_features(root, folder='finale'):
    with open(os.path.join(root, folder, 'modello.json'), encoding='utf-8') as f:
        info = json.load(f)
    return info.get('feature_version', 1) < 2


def ensemble_proba(components, df, it):
    """Media delle probabilità dei quattro pezzi (ognuno già simmetrico rispetto alle squadre), arrotondata."""
    import rete
    hgb, nets, cols = components
    parts = [E.predict_sym(hgb, df, cols)] + [rete.predict(net, df, it) for net in nets]
    return np.round(np.mean([p.astype(np.float64) for p in parts], axis=0), DECIMALS)


def _decode(proba, df, params):
    no_data = (df['n_casa'].values + df['n_ospiti'].values) == 0
    return M.decode(proba, df['period'].values, no_data, params)


def _params(root):
    return json.load(open(os.path.join(root, 'parametri_viterbi.json')))['params']


# --------------------------------------------------------------------------------------------------
# prevedi: il Barça
# --------------------------------------------------------------------------------------------------

def _possessions(codes, period):
    n = 0
    for half in (1, 2):
        z = codes[period == half]
        z = z[z != 3]
        if len(z):
            n += int(((z < 2) & np.r_[True, z[1:] != z[:-1]]).sum())
    return n


def _public_window(codes):
    sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
    import possession_metrics as pm
    gt = pm.label_codes(pd.read_csv(PUBLIC_GT)['label'].astype(str).values)
    r = pm.evaluate(gt, codes)
    return {'spellF1': round(float(r['spell_f1_mean']), 4), 'macroF1': round(float(r['macro_f1']), 4)}


def predict(root=FROZEN, name=NAME, write=False, report_public=True):
    t = time.time()
    for f in dati.input_files(dati.BARCA):
        if not os.path.exists(os.path.join(dati.C.fc.DATA_DIR, f)):
            sys.exit('manca data/{}: servono i file della gara in data/tactical/'.format(f))
    target = dati.table(dati.BARCA, legacy=legacy_features(root))
    target['fonte_tv'] = np.float32(0)              # il Barça ha una telecamera tattica, non le riprese tv
    proba = ensemble_proba(load_components(os.path.join(root, 'finale')), target, dati.BARCA)
    codes = _decode(proba, target, _params(root))
    out = os.path.join(RESULTS, name + '.csv')
    got = {'csv_sha256': _write_csv(out, codes),
           'conteggi': {k: int((LETTERS[codes] == k).sum()) for k in 'HADX'},
           'possessi': _possessions(codes, target['period'].values)}
    if report_public:
        got['finestra_pubblica'] = _public_window(codes)
    valid = subprocess.run([sys.executable, os.path.join(ROOT, 'evaluation', 'validate_submission.py'), out],
                           capture_output=True, text=True)
    if valid.returncode:
        raise ValueError('Submission non valida: ' + valid.stdout + valid.stderr)
    print('Barça: scritto {} ({:.0f} s), formato {}'.format(os.path.relpath(out, ROOT), time.time() - t,
                                                           'valido' if valid.returncode == 0 else 'NON VALIDO'))
    print('   possessi {}  conteggi {}'.format(got['possessi'], got['conteggi']))
    if report_public:
        print('   5 minuti pubblici (controllo finale, mai selezione): spellF1 {spellF1}  macroF1 {macroF1}'.format(
            **got['finestra_pubblica']))
    _compare('attesi.json', got, write, root, tuple(got.keys() - {'csv_sha256'}))
    return got


# --------------------------------------------------------------------------------------------------
# esterne: le 10 partite DFL + Metrica (e le 20 SkillCorner), ognuna dal modello che non l'ha vista
# --------------------------------------------------------------------------------------------------

def _external_proba(root, data):
    """Probabilità "fuori gruppo" di tutte le 30 partite, dai modelli dei 5 gruppi."""
    fold_of = E.folds(sorted(data))
    oof = {}
    for k in range(E.N_FOLDS):
        components = load_components(os.path.join(root, 'gruppi', 'gruppo{}'.format(k)))
        for it in sorted(data):
            if fold_of[it] == k:
                oof[it] = ensemble_proba(components, data[it], it)
    return oof


def _gap(codes, period):
    """Un frame X all'inizio del secondo tempo, come fa la dashboard: nessun possesso scavalca l'intervallo."""
    second = np.flatnonzero(period == 2)
    if len(second) == 0:
        return codes
    return np.concatenate([codes[:second[0]], np.array([3], dtype=codes.dtype), codes[second[0]:]])


def _score(df, codes):
    """spellF1 di una partita, calcolato esattamente come nella seconda pagina della dashboard."""
    per = df['period'].values
    return float(M.score(_gap(df['y'].values.astype(np.int8), per), _gap(codes.astype(np.int8), per)))


def _scores(oof, data, params):
    return {it: _score(data[it], _decode(oof[it], data[it], params)) for it in sorted(oof)}


def external(root=FROZEN, name=NAME, write=False, oof=None):
    t = time.time()
    legacy = legacy_features(root, os.path.join('gruppi', 'gruppo0'))
    dati.build_all(legacy=legacy)
    data = E.add_source_flag(dati.load_all(legacy=legacy))
    oof = oof or _external_proba(root, data)
    params = _params(root)
    hashes, per_match = {}, {}
    for it in sorted(oof):
        codes = _decode(oof[it], data[it], params)
        per_match[it[1]] = _score(data[it], codes)
        if it[0] in DM:                                  # la seconda pagina della dashboard
            hashes[it[1]] = _write_csv(os.path.join(EXT_RESULTS, name, it[1] + '.csv'), codes)
    by = {src: round(float(np.mean([per_match[i] for s, i in oof if s == src])), 4) for src in dati.SOURCES}
    ten = [per_match[i] for s, i in oof if s in DM]
    got = {'media10': round(float(np.mean(ten)), 4), 'per_fonte': by, 'peggiore10': round(min(ten), 4),
           'per_partita': {k: round(v, 4) for k, v in per_match.items()}, 'csv_sha256': hashes}
    print('Partite esterne: {} risposte in {} ({:.0f} s)'.format(len(hashes), os.path.relpath(
        os.path.join(EXT_RESULTS, name), ROOT), time.time() - t))
    print('   spellF1 medio sulle 10 DFL + Metrica: {}  (DFL {}, Metrica {}, peggiore {})  | SkillCorner {}'.format(
        got['media10'], by['dfl'], by['metrica'], got['peggiore10'], by['skillcorner']))
    _compare('attesi_esterne.json', got, write, root, ('media10', 'per_fonte', 'peggiore10'))
    return got


def _compare(fname, got, write, root, keys):
    """Confronta con i risultati attesi di questi pesi (root/fname) e, per una prova, anche con il riferimento."""
    path = os.path.join(root, fname)
    is_reference = os.path.abspath(root) == os.path.abspath(FROZEN)
    if write:
        json.dump(dict(got, ambiente=_versions()), open(path, 'w'), indent=1)
        print('   {} scritto'.format(os.path.relpath(path, ROOT)))
    elif os.path.exists(path):
        exp = json.load(open(path))
        if got['csv_sha256'] == exp['csv_sha256']:
            print('   RISULTATO: IDENTICO {} (stesse impronte SHA-256).'.format(
                'al riferimento' if is_reference else 'a quello atteso per questi pesi'))
        else:
            print('   RISULTATO: diverso da quello atteso per questi pesi.')
            for key in keys:
                print('      {:18s} atteso {}  ora {}'.format(key, exp[key], got[key]))
            print('   Con gli stessi pesi deve venire identico: controlla versioni e dati con "controlla".')
    ref_path = os.path.join(FROZEN, fname)
    if not is_reference and os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        print('   confronto con il riferimento {}:'.format(NAME))
        for key in keys:
            print('      {:18s} riferimento {}  questa prova {}'.format(key, ref[key], got[key]))


# --------------------------------------------------------------------------------------------------
# allena: tutta la ricetta da capo
# --------------------------------------------------------------------------------------------------

def _tune(oof, data, items):
    meta = {it: (data[it]['y'].values, data[it]['period'].values,
                 (data[it]['n_casa'].values + data[it]['n_ospiti'].values) == 0, None) for it in items}
    params, _ = E.tune(oof, meta, items, start=START, rounds=2)
    return params


def model_dir(name):
    """Dove stanno i pesi di una prova: data/cache_mv/modelli/<nome> (il riferimento sta in modelli_mv/congelato)."""
    if not name or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError('Nome modello non valido: usare lettere, numeri, _ e -')
    return FROZEN if name == NAME else os.path.join(dati.CACHE, 'modelli', name)


def train(name, freeze=False):
    import famiglie
    if name == NAME and not freeze:
        sys.exit('{} è il nome del riferimento: scegli un altro nome per la prova'.format(NAME))
    root = FROZEN if freeze else model_dir(name)
    if os.path.exists(os.path.join(root, 'gruppi')):
        shutil.rmtree(os.path.join(root, 'gruppi'))
    if os.path.exists(os.path.join(root, 'finale')):
        shutil.rmtree(os.path.join(root, 'finale'))
    E.WEIGHTS = True
    dati.build_all()
    data, items, cols, fold_of = famiglie.load()     # le 30 partite, con la colonna fonte_tv
    assert len(items) == 30, items
    t0 = time.time()
    print('1. Validazione: 5 gruppi di partite intere, ogni gruppo previsto da modelli che non l\'hanno visto')
    for k in range(E.N_FOLDS):
        fit_components(data, [it for it in items if fold_of[it] != k], cols,
                       os.path.join(root, 'gruppi', 'gruppo{}'.format(k)),
                       [SEED + 1000 * s + k for s in range(N_NETS)])
    oof = _external_proba(root, data)
    print('2. Parametri del Viterbi, regolati sulle 10 partite DFL + Metrica')
    dm = [it for it in items if it[0] in DM]
    params = _tune(oof, data, dm)
    nested = {}
    for k in range(E.N_FOLDS):                       # il punteggio "annidato": parametri regolati senza il gruppo k
        test = [it for it in dm if fold_of[it] == k]
        if test:
            p_k = _tune(oof, data, [it for it in dm if fold_of[it] != k])
            nested.update(_scores({it: oof[it] for it in test}, data, p_k))
    nested = round(float(np.mean(list(nested.values()))), 4)
    json.dump({'params': params, 'annidato': nested,
               'origine': 'Viterbi regolato (ricerca per coordinate, 2 giri, partendo da START di riproduci.py) '
                          'sulle probabilità fuori gruppo delle 10 partite DFL + Metrica; annidato = spellF1 medio '
                          'con i parametri regolati senza il gruppo giudicato',
               'unita': 'turn/stop/rest: costi dei cambi; m_pos/m_dead: durate minime in frame (25 al secondo); '
                        'bias_d: spinta verso D'},
              open(os.path.join(root, 'parametri_viterbi.json'), 'w'), indent=1)
    print('   parametri {}  annidato {}'.format(params, nested))
    print('3. Modello finale per il Barça, allenato su tutte e 30 le partite')
    fit_components(data, items, cols, os.path.join(root, 'finale'), [SEED + 1000 * s for s in range(N_NETS)])
    print('4. Risultati ({:.0f} min)'.format((time.time() - t0) / 60))
    out_name = NAME if freeze else name
    external(root, out_name, write=True, oof=oof)
    predict(root, out_name, write=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cosa', choices=['controlla', 'prevedi', 'esterne', 'allena'])
    ap.add_argument('--solo-barca', action='store_true', help='controlla: solo i file della gara (basta per prevedi)')
    ap.add_argument('--nome', help='allena: nome della prova (cartella data/cache_mv/modelli/NOME e nome in dashboard)')
    ap.add_argument('--modello', default=NAME,
                    help='prevedi/esterne: i pesi di quale prova usare (di base il riferimento %(default)s)')
    ap.add_argument('--congela', action='store_true', help=argparse.SUPPRESS)   # solo per chi aggiorna il riferimento
    a = ap.parse_args()
    if a.cosa == 'controlla':
        sys.exit(0 if check(write=a.congela, only_barca=a.solo_barca) else 1)
    elif a.cosa in ('prevedi', 'esterne'):
        root = model_dir(a.modello)
        if not os.path.isdir(os.path.join(root, 'finale' if a.cosa == 'prevedi' else 'gruppi')):
            sys.exit('non trovo i pesi di "{}" in {}'.format(a.modello, os.path.relpath(root, ROOT)))
        (predict if a.cosa == 'prevedi' else external)(root, a.modello, write=a.congela)
    else:
        if not a.nome and not a.congela:
            sys.exit('serve un nome per la prova: --nome NOME (per esempio --nome ste_prova1)')
        train(a.nome, freeze=a.congela)
