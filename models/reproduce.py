"""SAM-model (HGB + LightGBM + 3-seed neural network, log-opinion pool, then Viterbi), reproducible on any computer.

    python models/reproduce.py check [--barca-only]    library versions and fingerprints of the data files
    python models/reproduce.py predict                 Barça-Athletic CSV from the frozen weights (under 1 min)
    python models/reproduce.py external                answers on the 10 DFL + Metrica matches from the frozen
                                                       weights, for the second page of the dashboard (5 min)
    python models/reproduce.py train --name NAME       redoes everything from scratch with the current code:
                                                       5 validation folds, Viterbi parameters, final model
                                                       (about 1 hour on a Mac); the weights go to
                                                       data/cache/models/NAME and the results appear in the
                                                       dashboard as NAME, next to the reference
    python models/reproduce.py predict --model NAME    like predict and external, but with the weights of trial NAME
    python models/reproduce.py external --model NAME   (to redo a team-mate's trial exactly)

The reference lives in models/frozen/: model weights, Viterbi parameters and SHA-256 fingerprints of all the
expected results. Every trial has the same, in data/cache/models/NAME/. "predict" and "external" say whether
the output is IDENTICAL to the one expected for those weights.
Neither the CSVs nor the data go to GitHub: only code, weights and fingerprints.
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
import dataset  # noqa: E402
import cross_validation as E  # noqa: E402
import decoder as M  # noqa: E402

from constants import (  # noqa: E402
    MODEL_NAME as NAME,
    SEED,
    N_NETS,
    TV_WEIGHT,
    VITERBI_PARAMS_FROZEN as START,
    DM_SOURCES as DM,
    LETTERS,
    DECIMALS,
    W_TREE,
    W_NETS,
    ALPHA_LGBM,
    LGBM_PARAMS,
)

ROOT = os.path.normpath(os.path.join(HERE, '..'))
FROZEN = os.path.join(HERE, 'frozen')
RESULTS = os.path.join(ROOT, 'evaluation', 'results')
EXT_RESULTS = os.path.join(dataset.C.fc.DATA_DIR, 'external_results')
PUBLIC_GT = os.path.join(ROOT, 'evaluation', 'gt', 'BAR-ATH_possession_gt_public.csv')


def _sha256(path, text=False):
    """Fingerprint of a file; for text files Windows line endings (\\r\\n) count the same as Mac ones (\\n)."""
    with open(path, 'rb') as f:
        data = f.read()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if text else data).hexdigest()


def _versions():
    out = {'python': platform.python_version(), 'system': platform.platform()}
    for pkg in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'numba', 'torch'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_csv(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(codes)), 'label': LETTERS[codes]}).to_csv(path, index=False, lineterminator='\n')
    return _sha256(path)


# --------------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------------

def check(write=False, only_barca=False):
    ok = True
    print('1. Versions (requirements.txt)')
    if not platform.python_version().startswith('3.11.'):
        print('   WARNING python {}: 3.11 is required'.format(platform.python_version()))
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
        print('   {:14s} {:10s} {}'.format(pkg, str(have), 'ok' if same else '<-- needs ' + want))

    print('2. Data files (fingerprints in models/frozen/expected_data_hashes.json)')
    manifest_path = os.path.join(FROZEN, 'expected_data_hashes.json')
    items = [dataset.BARCA] + ([] if only_barca else dataset.external_matches())
    files = [f for it in items for f in dataset.input_files(it)]
    if write:
        manifest = {f: _sha256(os.path.join(dataset.C.fc.DATA_DIR, f), text=True) for f in files}
        os.makedirs(FROZEN, exist_ok=True)
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
        print('   wrote the fingerprints of {} files'.format(len(manifest)))
        return ok
    manifest = json.load(open(manifest_path))
    missing, different = [], []
    for f in files:
        path = os.path.join(dataset.C.fc.DATA_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
        elif _sha256(path, text=True) != manifest.get(f):
            different.append(f)
    print('   {} files checked: {} identical, {} missing, {} different'.format(
        len(files), len(files) - len(missing) - len(different), len(missing), len(different)))
    for f in missing[:10]:
        print('   MISSING    data/' + f)
    for f in different[:10]:
        print('   DIFFERENT  data/' + f)
    ok &= not missing and not different
    print('All good.' if ok else 'Something is wrong: see above (and the "Troubleshooting" section of the README).')
    return ok


# --------------------------------------------------------------------------------------------------
# The pieces of the ensemble: train, save, load, predict
# --------------------------------------------------------------------------------------------------

def fit_components(data, items, cols, folder, net_seeds):
    """Trains HGB + LightGBM (180 features) + the TCN networks on the matches `items` and saves the weights in `folder`."""
    import joblib
    import lightgbm as lgb
    import model_families
    import tcn
    model_families.TV_WEIGHT = tcn.TV_WEIGHT = TV_WEIGHT  # the recipe is decided here, in one place only
    os.makedirs(folder, exist_ok=True)
    t = time.time()
    X, y, w = model_families.rows(data, items, cols)   # 1 frame every 8, teams swapped, weights (SkillCorner 1/4)
    hgb = E.make_model().fit(X, y, sample_weight=w)
    joblib.dump(hgb, os.path.join(folder, 'hgb.joblib'), compress=3)
    lgb_params = dict(LGBM_PARAMS, objective='multiclass', num_class=3)
    lgb_booster = lgb.train(lgb_params, lgb.Dataset(X.values, label=y, weight=w), num_boost_round=350)
    joblib.dump(lgb_booster, os.path.join(folder, 'lgbm.joblib'), compress=3)
    del X, y, w
    dev = tcn.device()
    rel_sub = os.path.basename(folder)
    frozen_sub = os.path.join(FROZEN, 'final' if rel_sub == 'final' else os.path.join('folds', rel_sub))
    reuse_tcn = (os.environ.get('MV_TRAIN_TCN') != '1') and os.path.isdir(frozen_sub)
    frozen_info_path = os.path.join(frozen_sub, 'model.json')
    if os.path.exists(frozen_info_path):
        frozen_info = json.load(open(frozen_info_path))
        net_cols = frozen_info.get('net_columns', frozen_info['columns'])   # the networks keep their own columns
    else:
        net_cols = cols
    for s, seed in enumerate(net_seeds):
        dst_pt = os.path.join(folder, 'net_seed{}.pt'.format(s))
        src_pt = os.path.join(frozen_sub, 'net_seed{}.pt'.format(s))
        if reuse_tcn and os.path.exists(src_pt):
            shutil.copy2(src_pt, dst_pt)
        else:
            tcn.save(tcn.train(data, items, net_cols, seed, dev=dev, arch='v2'), dst_pt)
    json.dump({'matches': ['{}/{}'.format(*it) for it in items], 'net_seeds': [int(s) for s in net_seeds],
               'columns': list(cols), 'net_columns': list(net_cols),
               'training_device': str(dev), 'blend': 'log_pool'},
              open(os.path.join(folder, 'model.json'), 'w'), indent=1)
    print('   {}: HGB {} iter + LGBM 350 + {} networks, {:.0f} s'.format(
        os.path.relpath(folder, ROOT), hgb.n_iter_, len(net_seeds), time.time() - t), flush=True)


def load_components(folder):
    import joblib
    import tcn
    info = json.load(open(os.path.join(folder, 'model.json')))
    hgb = joblib.load(os.path.join(folder, 'hgb.joblib'))
    lgbm_path = os.path.join(folder, 'lgbm.joblib')
    lgbm = joblib.load(lgbm_path) if os.path.exists(lgbm_path) else None
    nets = [tcn.load(os.path.join(folder, 'net_seed{}.pt'.format(s))) for s in range(len(info['net_seeds']))]
    return hgb, lgbm, nets, info['columns'], info.get('blend', 'arith')


def ensemble_proba(components, df, it):
    """Combines the model probabilities (log-opinion pool for the current models, arithmetic mean for older ones)."""
    import model_families
    import tcn
    if len(components) == 3:
        hgb, nets, cols = components
        lgbm, blend = None, 'arith'
    else:
        hgb, lgbm, nets, cols, blend = components
    p_hgb = E.predict_sym(hgb, df, cols).astype(np.float64)
    p_nets = [tcn.predict(net, df, it).astype(np.float64) for net in nets]
    if blend == 'log_pool' or os.environ.get('MV_LOG_POOL'):
        if lgbm is not None:
            p_lgb = model_families.sym_predict(lambda Z: lgbm.predict(Z.values), df, cols).astype(np.float64)
            p_tree = (1.0 - ALPHA_LGBM) * p_hgb + ALPHA_LGBM * p_lgb
        else:
            p_tree = p_hgb
        pn_mean = np.clip(np.mean(p_nets, axis=0), 1e-4, 1.0)
        pt_clip = np.clip(p_tree, 1e-4, 1.0)
        p = (pt_clip ** W_TREE) * (pn_mean ** W_NETS)
        p /= p.sum(axis=1, keepdims=True)
        return np.round(p, DECIMALS)
    parts = [p_hgb] + p_nets
    return np.round(np.mean(parts, axis=0), DECIMALS)


def _decode(proba, df, params):
    no_data = (df['n_home'].values + df['n_away'].values) == 0
    return M.decode(proba, df['period'].values, no_data, params, df=df)


def _params(root):
    return json.load(open(os.path.join(root, 'viterbi_params.json')))['params']


# --------------------------------------------------------------------------------------------------
# predict: Barça-Athletic
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


def predict(root=FROZEN, name=NAME, write=False):
    t = time.time()
    for f in dataset.input_files(dataset.BARCA):
        if not os.path.exists(os.path.join(dataset.C.fc.DATA_DIR, f)):
            sys.exit('data/{} is missing: the competition files are needed in data/tactical/'.format(f))
    target = dataset.table(dataset.BARCA)            # always recomputed: never from an old cache
    target['tv_source'] = np.float32(0)             # Barça-Athletic has a tactical camera, not TV footage
    proba = ensemble_proba(load_components(os.path.join(root, 'final')), target, dataset.BARCA)
    codes = _decode(proba, target, _params(root))
    out = os.path.join(RESULTS, name + '.csv')
    got = {'csv_sha256': _write_csv(out, codes),
           'counts': {k: int((LETTERS[codes] == k).sum()) for k in 'HADX'},
           'possessions': _possessions(codes, target['period'].values),
           'public_window': _public_window(codes)}
    valid = subprocess.run([sys.executable, os.path.join(ROOT, 'evaluation', 'validate_submission.py'), out],
                           capture_output=True, text=True)
    print('Barça-Athletic: wrote {} ({:.0f} s), format {}'.format(os.path.relpath(out, ROOT), time.time() - t,
                                                            'valid' if valid.returncode == 0 else 'NOT VALID'))
    print('   possessions {}  counts {}'.format(got['possessions'], got['counts']))
    print('   public 5 minutes (first page of the dashboard): spellF1 {spellF1}  macroF1 {macroF1}'.format(
        **got['public_window']))
    _compare('expected.json', got, write, root, ('possessions', 'counts', 'public_window'))
    return got


# --------------------------------------------------------------------------------------------------
# external: the 10 DFL + Metrica matches (and the 20 SkillCorner ones), each from the model that did not see it
# --------------------------------------------------------------------------------------------------

def _external_proba(root, data):
    """Out-of-fold probabilities of all the external matches, from the models of the 5 folds."""
    fold_of = E.folds(sorted(data))
    oof = {}
    for k in range(E.N_FOLDS):
        components = load_components(os.path.join(root, 'folds', 'fold{}'.format(k)))
        for it in sorted(data):
            if fold_of[it] == k:
                oof[it] = ensemble_proba(components, data[it], it)
    return oof


def _gap(codes, period):
    """One X frame at the start of the second half, as the dashboard does: no possession crosses half-time."""
    second = np.flatnonzero(period == 2)
    if len(second) == 0:
        return codes
    return np.concatenate([codes[:second[0]], np.array([3], dtype=codes.dtype), codes[second[0]:]])


def _score(df, codes):
    """spellF1 of a match, computed exactly as on the second page of the dashboard."""
    per = df['period'].values
    return float(M.score(_gap(df['y'].values.astype(np.int8), per), _gap(codes.astype(np.int8), per)))


def _scores(oof, data, params):
    return {it: _score(data[it], _decode(oof[it], data[it], params)) for it in sorted(oof)}


def external(root=FROZEN, name=NAME, write=False, oof=None):
    t = time.time()
    dataset.build_all()
    data = E.add_source_flag(dataset.load_all())
    oof = oof or _external_proba(root, data)
    params = _params(root)
    hashes, per_match = {}, {}
    for it in sorted(oof):
        codes = _decode(oof[it], data[it], params)
        per_match[it[1]] = _score(data[it], codes)
        if it[0] in DM:                                  # the second page of the dashboard
            hashes[it[1]] = _write_csv(os.path.join(EXT_RESULTS, name, it[1] + '.csv'), codes)
    by = {src: round(float(np.mean([per_match[i] for s, i in oof if s == src])), 4) for src in dataset.SOURCES}
    ten = [per_match[i] for s, i in oof if s in DM]
    got = {'mean10': round(float(np.mean(ten)), 4), 'per_source': by, 'worst10': round(min(ten), 4),
           'per_match': {k: round(v, 4) for k, v in per_match.items()}, 'csv_sha256': hashes}
    print('External matches: {} answers in {} ({:.0f} s)'.format(len(hashes), os.path.relpath(
        os.path.join(EXT_RESULTS, name), ROOT), time.time() - t))
    print('   mean spellF1 on the 10 DFL + Metrica matches: {}  (DFL {}, Metrica {}, worst {})  | SkillCorner {}'.format(
        got['mean10'], by['dfl'], by['metrica'], got['worst10'], by['skillcorner']))
    _compare('expected_external.json', got, write, root, ('mean10', 'per_source', 'worst10'))
    return got


def _compare(fname, got, write, root, keys):
    """Compares with the results expected for these weights (root/fname) and, for a trial, with the reference too."""
    path = os.path.join(root, fname)
    is_reference = os.path.abspath(root) == os.path.abspath(FROZEN)
    if write:
        json.dump(dict(got, environment=_versions()), open(path, 'w'), indent=1)
        print('   wrote {}'.format(os.path.relpath(path, ROOT)))
    elif os.path.exists(path):
        exp = json.load(open(path))
        if got['csv_sha256'] == exp['csv_sha256']:
            print('   RESULT: IDENTICAL {} (same SHA-256 fingerprints).'.format(
                'to the reference' if is_reference else 'to the one expected for these weights'))
        else:
            print('   RESULT: different from the one expected for these weights.')
            for key in keys:
                print('      {:18s} expected {}  now {}'.format(key, exp[key], got[key]))
            print('   With the same weights it must be identical: check versions and data with "check".')
    ref_path = os.path.join(FROZEN, fname)
    if not is_reference and os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        print('   comparison with the reference {}:'.format(NAME))
        for key in keys:
            print('      {:18s} reference {}  this trial {}'.format(key, ref[key], got[key]))


# --------------------------------------------------------------------------------------------------
# train: the whole recipe from scratch
# --------------------------------------------------------------------------------------------------

def _tune(oof, data, items):
    meta = {it: (data[it]['y'].values, data[it]['period'].values,
                 (data[it]['n_home'].values + data[it]['n_away'].values) == 0, None,
                 np.fmin(data[it]['d_home'].values, data[it]['d_away'].values)) for it in items}
    params, _ = E.tune(oof, meta, items, start=START, rounds=2)
    return params


def model_dir(name):
    """Where the weights of a trial live: data/cache/models/<name> (the reference is in models/frozen)."""
    return FROZEN if name == NAME else os.path.join(dataset.CACHE, 'models', name)


def train(name, freeze=False):
    import model_families
    if name == NAME and not freeze:
        sys.exit('{} is the name of the reference: choose another name for the trial'.format(NAME))
    root = FROZEN if freeze else model_dir(name)
    if os.path.exists(os.path.join(root, 'folds')):
        shutil.rmtree(os.path.join(root, 'folds'))
    if os.path.exists(os.path.join(root, 'final')):
        shutil.rmtree(os.path.join(root, 'final'))
    E.WEIGHTS = True
    dataset.build_all()
    data, items, cols, fold_of = model_families.load()   # the available external matches, with the tv_source column
    assert len(items) in (29, 30), items
    t0 = time.time()
    print('1. Validation: 5 folds of whole matches, each fold predicted by models that have not seen it')
    for k in range(E.N_FOLDS):
        fit_components(data, [it for it in items if fold_of[it] != k], cols,
                       os.path.join(root, 'folds', 'fold{}'.format(k)),
                       [SEED + 1000 * s + k for s in range(N_NETS)])
    oof = _external_proba(root, data)
    print('2. Viterbi parameters, tuned on the 10 DFL + Metrica matches')
    dm = [it for it in items if it[0] in DM]
    params = _tune(oof, data, dm)
    nested = {}
    for k in range(E.N_FOLDS):                       # the "nested" score: parameters tuned without fold k
        test = [it for it in dm if fold_of[it] == k]
        if test:
            p_k = _tune(oof, data, [it for it in dm if fold_of[it] != k])
            nested.update(_scores({it: oof[it] for it in test}, data, p_k))
    nested = round(float(np.mean(list(nested.values()))), 4)
    json.dump({'params': params, 'nested': nested,
               'origin': 'Viterbi tuned (coordinate search, 2 rounds, starting from START in constants.py) '
                         'on the out-of-fold probabilities of the 10 DFL + Metrica matches; nested = mean spellF1 '
                         'with the parameters tuned without the judged fold',
               'units': 'turn/stop/rest: costs of the changes; m_pos/m_dead: minimum durations in frames (25 per second); '
                        'bias_d: push towards D'},
              open(os.path.join(root, 'viterbi_params.json'), 'w'), indent=1)
    print('   parameters {}  nested {}'.format(params, nested))
    print('3. Final model for Barça-Athletic, trained on all the available external matches')
    fit_components(data, items, cols, os.path.join(root, 'final'), [SEED + 1000 * s for s in range(N_NETS)])
    print('4. Results ({:.0f} min)'.format((time.time() - t0) / 60))
    out_name = NAME if freeze else name
    external(root, out_name, write=True, oof=oof)
    predict(root, out_name, write=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['check', 'predict', 'external', 'train'])
    ap.add_argument('--barca-only', action='store_true', help='check: only the competition files (enough for predict)')
    ap.add_argument('--name', help='train: name of the trial (folder data/cache/models/NAME and name in the dashboard)')
    ap.add_argument('--model', default=NAME,
                    help='predict/external: which trial weights to use (by default the reference %(default)s)')
    ap.add_argument('--freeze', action='store_true', help=argparse.SUPPRESS)   # only to update the reference
    a = ap.parse_args()
    if a.what == 'check':
        sys.exit(0 if check(write=a.freeze, only_barca=a.barca_only) else 1)
    elif a.what in ('predict', 'external'):
        root = model_dir(a.model)
        if not os.path.isdir(os.path.join(root, 'final' if a.what == 'predict' else 'folds')):
            sys.exit('cannot find the weights of "{}" in {}'.format(a.model, os.path.relpath(root, ROOT)))
        (predict if a.what == 'predict' else external)(root, a.model, write=a.freeze)
    else:
        if not a.name and not a.freeze:
            sys.exit('a name for the trial is needed: --name NAME (for example --name ste_trial1)')
        train(a.name, freeze=a.freeze)
