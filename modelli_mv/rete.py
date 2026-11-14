"""La rete neurale convoluzionale sulla sequenza: 5 frame al secondo, ±25 s di contesto prima e dopo ogni momento.

    train(data, items, cols, seed)   allena su quelle partite; restituisce il modello (rete + normalizzazione)
    predict(model, df, it)           (T, 3) probabilità H, A, D per ogni frame della partita
    save(model, path) / load(path)   pesi su disco (file .pt di PyTorch, pochi MB)

Architettura: convoluzione 1×1 (284 -> 96 canali), 7 blocchi residui di convoluzioni dilatate (dilatazione
1, 2, ..., 64; BatchNorm, GELU, Dropout 0,15), convoluzione 1×1 (96 -> 3). L'input sono le caratteristiche
normalizzate (mediana e scarto interquartile delle partite di allenamento) più un indicatore per ogni valore mancante.

Dove gira: scheda grafica NVIDIA (cuda) o Apple (mps) se c'è, altrimenti processore. Si può imporre con la
variabile d'ambiente MV_DEVICE (cpu, cuda, mps). La previsione dai pesi salvati si fa sempre sul processore.
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caratteristiche as C  # noqa: E402
import esperimento as E  # noqa: E402
import constants as K  # noqa: E402

STRIDE = K.STRIDE        # 5 frame al secondo: la rete guarda la sequenza a 0,2 s di passo
CROP = K.CROP            # tratti di 60 s per allenare
EPOCHS = int(os.environ.get('MV_EPOCHS', K.EPOCHS))
TV_WEIGHT = K.TV_WEIGHT  # peso dei frame SkillCorner


def device():
    choice = os.environ.get('MV_DEVICE')
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Block(nn.Module):
    """Convoluzione "dilatata": guarda frame sempre più lontani, avanti e indietro, con un collegamento diretto."""
    def __init__(self, c, d):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.norm = nn.BatchNorm1d(c)
        self.act = nn.GELU()
        self.drop = nn.Dropout(K.DROPOUT)

    def forward(self, x):
        return x + self.drop(self.act(self.norm(self.conv(x))))


class Net(nn.Module):
    """Rete convoluzionale non causale: ogni momento vede ±127 passi (±25 s) intorno a sé."""
    def __init__(self, n_in, c=K.CHANNELS):
        super().__init__()
        self.inp = nn.Conv1d(n_in, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in K.DILATIONS])
        self.out = nn.Conv1d(c, 3, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)


_RAW = {}


def _raw(df, it, cols, *, legacy=False):
    """Una volta per partita: i frame a 5 al secondo di ogni tempo, diretti e a squadre scambiate (non normalizzati)."""
    # A match name does not identify its data: augmentation views and refreshed
    # caches share it. Do not retain mutable DataFrames in a process-wide cache.
    return raw_sequences(df, cols, TV_WEIGHT if it[0] == 'skillcorner' else 1.0,
                         legacy=legacy)


def raw_sequences(df, cols, source_weight, *, legacy=False):
    per = df['period'].values
    y_all = df['y'].values.astype(np.int64)
    w_all = E.short_spell_weights(df['y'].values, per) * source_weight
    out = []
    if legacy:
        segments = [np.flatnonzero(per == half) for half in (1, 2)]
        segments = [idx for idx in segments if len(idx) >= CROP * STRIDE]
    else:
        active = per.copy()
        if {'n_casa', 'n_ospiti', 'palla_vista'}.issubset(df.columns):
            empty = ((df['n_casa'].fillna(0) + df['n_ospiti'].fillna(0)) == 0)
            empty &= df['palla_vista'].fillna(0) == 0
            active[empty.to_numpy()] = 0
        segments = [np.arange(a, b) for a, b in C._segments(active)]
    for idx in segments:
        sel = idx[::STRIDE]
        sub = df[cols].iloc[sel]
        xo = sub.values.astype(np.float32)
        xs = C.swap_teams(sub)[cols].values.astype(np.float32)
        y = y_all[sel]
        ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
        out.append((xo, xs, y, ys, w_all[sel].astype(np.float32), sel))
    return out


def _seqs(df, it, cols, med, iqr, swap=False, *, legacy=False):
    """Per ogni tempo: (caratteristiche normalizzate + indicatori dei mancanti, etichette, pesi, frame)."""
    out = []
    for xo, xs, y, ys, w, sel in _raw(df, it, cols, legacy=legacy):
        x = ((xs if swap else xo) - med) / iqr
        miss = ~np.isfinite(x)
        x[miss] = np.nan
        x = np.clip(np.nan_to_num(x, nan=0.0), *K.NET_FEATURE_CLIP)
        x = np.concatenate([x, miss.astype(np.float32)], axis=1).astype(np.float32)
        out.append((x, ys if swap else y, w, sel))
    return out


def normalizer(samples):
    if not samples or not any(len(sample) for sample in samples):
        raise ValueError('nessuna riga di training per la normalizzazione')
    allX = np.concatenate(samples)
    allX = np.where(np.isfinite(allX), allX, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(allX, axis=0)
        iqr = np.nanpercentile(allX, 75, axis=0) - np.nanpercentile(allX, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, iqr


def training_crop(x, y, w, start, length=CROP, context=None):
    """Fixed-size supervised centre with an unsupervised receptive-field halo.

    Crops in the middle of a half see the same context as full-sequence inference.
    Short tracking segments are padded; padding and halo never contribute labels
    to the loss. No feature window crosses a tracking discontinuity.
    """
    context = sum(K.DILATIONS) if context is None else int(context)
    if len(x) != len(y) or len(y) != len(w) or not len(y):
        raise ValueError('sequenza vuota o lunghezze x/y/w incoerenti')
    if start < 0 or start >= len(y) or length < 1 or context < 0:
        raise ValueError('crop o contesto non valido')
    size = length + 2 * context
    xc = np.full((size, x.shape[1]), np.nan, dtype=np.float32)
    yc = np.full(size, K.X, dtype=np.int64)
    wc = np.zeros(size, dtype=np.float32)
    lo, hi = max(0, start - context), min(len(y), start + length + context)
    offset = lo - (start - context)
    xc[offset:offset + hi - lo] = x[lo:hi]
    stop = min(len(y), start + length)
    n = stop - start
    yc[context:context + n] = y[start:stop]
    wc[context:context + n] = w[start:stop]
    return xc, yc, wc


def fit_batches(cols, seed, dev, med, iqr, per_epoch, batch, *, raw=False, epochs=None, progress=None):
    dev = torch.device(dev)
    torch.manual_seed(seed)
    epochs = EPOCHS if epochs is None else epochs
    if epochs < 1 or per_epoch < 1:
        raise ValueError('epoche e batch per epoca devono essere positivi')
    net = Net(2 * len(cols)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=K.LR, weight_decay=K.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=epochs * per_epoch)
    rng = np.random.default_rng(seed)
    mixed_precision = raw and dev.type == 'cuda' and torch.cuda.is_bf16_supported()
    if raw:
        median = torch.as_tensor(med, device=dev, dtype=torch.float32)
        scale = torch.as_tensor(iqr, device=dev, dtype=torch.float32)
    for epoch in range(epochs):
        net.train()
        for _ in range(per_epoch):
            xb, yb, wb = batch(rng)
            xb = torch.from_numpy(np.stack(xb)).to(dev)
            yb = torch.from_numpy(np.stack(yb)).to(dev)
            wb = torch.from_numpy(np.stack(wb)).to(dev)
            if raw:
                xb = (xb - median) / scale
                miss = ~torch.isfinite(xb)
                xb = torch.cat((xb.masked_fill(miss, 0.0).clamp(*K.NET_FEATURE_CLIP), miss.float()), dim=-1)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=mixed_precision):
                logits = net(xb).float()
            valid = ((yb >= 0) & (yb < 3)).float() * wb
            if not torch.isfinite(wb).all() or (wb < 0).any():
                raise ValueError('pesi di training non finiti o negativi')
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.clamp(min=0, max=2).reshape(-1), reduction='none')
            loss = (loss * valid.reshape(-1)).sum() / valid.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), K.GRAD_CLIP)
            opt.step()
            sched.step()
        if progress is not None:
            progress(epoch + 1, epochs)
    net.eval()
    return {'net': net, 'med': med, 'iqr': iqr, 'cols': list(cols), 'seed': int(seed),
            'per_epoch': per_epoch, 'inference_version': 2}


def train(data, items, cols, seed, dev=None):
    """Allena la rete sulle partite `items` con il campionamento della ricetta congelata."""
    dev = dev or device()
    samples = []
    for it in items:
        df = data[it]
        valid = (df['period'] > 0) & df['y'].between(0, 2)
        sub = df.loc[valid, cols].iloc[::K.NORM_SUBSAMPLE_STEP]
        samples.extend((sub.to_numpy(), C.swap_teams(sub)[cols].to_numpy()))
    med, iqr = normalizer(samples)
    seqs = []
    for it in items:
        seqs += _seqs(data[it], it, cols, med, iqr) + _seqs(data[it], it, cols, med, iqr, swap=True)
    lens = np.array([len(q[0]) for q in seqs], dtype=float)
    if not len(seqs):
        raise ValueError('nessuna sequenza di training valida')
    probability = lens / lens.sum()
    per_epoch = 3 if os.environ.get('MV_SMOKE') else max(1, int(lens.sum()) // (CROP * K.BATCH_SIZE))

    def batch(rng):
        xb, yb, wb = [], [], []
        for si in rng.choice(len(seqs), size=K.BATCH_SIZE, p=probability):
            x, y, w, _ = seqs[si]
            a = rng.integers(0, max(1, len(x) - CROP + 1))
            xc, yc, wc = training_crop(x, y, w, a)
            # _seqs already normalised the features; true segment padding must
            # be finite before it enters the network.
            xb.append(np.nan_to_num(xc, nan=0.0))
            yb.append(yc)
            wb.append(wc)
        return xb, yb, wb

    return fit_batches(cols, seed, dev, med, iqr, per_epoch, batch)


def predict(model, df, it):
    """(T, 3) probabilità H, A, D: media della previsione diretta e di quella a squadre scambiate (rimesse a posto)."""
    net, med, iqr, cols = model['net'], model['med'], model['iqr'], model['cols']
    dev = next(net.parameters()).device
    T = len(df)
    legacy = model.get('inference_version', 1) < 2
    net.eval()
    probs = []
    for swap in (False, True):
        full = np.full((T, 3), 1 / 3, dtype=np.float32)
        for x, _, _, sel in _seqs(df, it, cols, med, iqr, swap=swap, legacy=legacy):
            with torch.no_grad():
                p = torch.softmax(net(torch.from_numpy(x[None]).to(dev)), -1)[0].cpu().numpy()
            if swap:
                p = p[:, [1, 0, 2]]
            a, b = sel[0], sel[-1] + 1
            if not legacy:
                # Include the unsampled tail, but never cross a period or a
                # no-tracking break. Repeat the final probability there.
                per = df['period'].to_numpy()
                while b < min(T, sel[-1] + STRIDE) and per[b] == per[a]:
                    if {'n_casa', 'n_ospiti', 'palla_vista'}.issubset(df.columns):
                        row = df.iloc[b]
                        if not (row['n_casa'] + row['n_ospiti'] > 0 or row['palla_vista'] > 0):
                            break
                    b += 1
            grid = np.arange(a, b)
            full[a:b] = np.stack([np.interp(grid, sel, p[:, c]) for c in range(3)], axis=1)
        probs.append(full)
    return (probs[0] + probs[1]) / 2


def save(model, path):
    net = model['net']
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()},
                'n_in': int(net.inp.in_channels),
                'med': torch.from_numpy(np.ascontiguousarray(model['med'])),
                'iqr': torch.from_numpy(np.ascontiguousarray(model['iqr'])),
                'cols': list(model['cols']), 'seed': int(model['seed']),
                'inference_version': int(model.get('inference_version', 1)),
                'channels': int(net.inp.out_channels), 'dilations': list(K.DILATIONS)}, path)


def load(path):
    """Il modello salvato, sul processore e in modalità previsione."""
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if obj.get('dilations', list(K.DILATIONS)) != list(K.DILATIONS):
        raise ValueError('dilatazioni del checkpoint diverse dalla configurazione della rete')
    net = Net(obj['n_in'], c=obj.get('channels', K.CHANNELS))
    net.load_state_dict(obj['state_dict'])
    net.eval()
    return {'net': net, 'med': obj['med'].numpy(), 'iqr': obj['iqr'].numpy(), 'cols': obj['cols'],
            'seed': obj['seed'], 'inference_version': obj.get('inference_version', 1)}
