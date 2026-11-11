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


def _raw(df, it, cols):
    """Una volta per partita: i frame a 5 al secondo di ogni tempo, diretti e a squadre scambiate (non normalizzati)."""
    if it in _RAW:
        return _RAW[it]
    per = df['period'].values
    y_all = df['y'].values.astype(np.int64)
    w_all = E.short_spell_weights(df['y'].values, per) * (TV_WEIGHT if it[0] == 'skillcorner' else 1.0)
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(per == half)
        if len(idx) < CROP * STRIDE:
            continue
        sel = idx[::STRIDE]
        sub = df[cols].iloc[sel]
        xo = sub.values.astype(np.float32)
        xs = C.swap_teams(sub)[cols].values.astype(np.float32)
        y = y_all[sel]
        ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
        out.append((xo, xs, y, ys, w_all[sel].astype(np.float32), sel))
    _RAW[it] = out
    return out


def _seqs(df, it, cols, med, iqr, swap=False):
    """Per ogni tempo: (caratteristiche normalizzate + indicatori dei mancanti, etichette, pesi, frame)."""
    out = []
    for xo, xs, y, ys, w, sel in _raw(df, it, cols):
        x = ((xs if swap else xo) - med) / iqr
        miss = np.isnan(x)
        x = np.clip(np.nan_to_num(x, nan=0.0), *K.NET_FEATURE_CLIP)
        x = np.concatenate([x, miss.astype(np.float32)], axis=1).astype(np.float32)
        out.append((x, ys if swap else y, w, sel))
    return out


def train(data, items, cols, seed, dev=None):
    """Allena la rete sulle partite `items`. Restituisce {'net', 'med', 'iqr', 'cols', 'seed', 'per_epoch'}."""
    torch.manual_seed(seed)
    dev = dev or device()
    allX = np.concatenate([data[it][cols].values[::K.NORM_SUBSAMPLE_STEP] for it in items])
    med = np.nanmedian(allX, axis=0)
    iqr = np.nanpercentile(allX, 75, axis=0) - np.nanpercentile(allX, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    seqs = []
    for it in items:
        seqs += _seqs(data[it], it, cols, med, iqr) + _seqs(data[it], it, cols, med, iqr, swap=True)

    net = Net(seqs[0][0].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=K.LR, weight_decay=K.WEIGHT_DECAY)
    lens = np.array([len(q[0]) for q in seqs], dtype=float)
    per_epoch = max(1, int(lens.sum()) // (CROP * K.BATCH_SIZE))
    if os.environ.get('MV_SMOKE'):
        per_epoch = 3
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=EPOCHS * per_epoch)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        net.train()
        for _ in range(per_epoch):
            xb, yb, wb = [], [], []
            for si in rng.choice(len(seqs), size=K.BATCH_SIZE, p=lens / lens.sum()):
                x, y, w, _ = seqs[si]
                a = rng.integers(0, len(x) - CROP + 1)
                xb.append(x[a:a + CROP])
                yb.append(y[a:a + CROP])
                wb.append(w[a:a + CROP])
            xb = torch.from_numpy(np.stack(xb)).to(dev)
            yb = torch.from_numpy(np.stack(yb)).to(dev)
            wb = torch.from_numpy(np.stack(wb)).to(dev)
            logits = net(xb)
            valid = (yb < 3).float() * wb
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.clamp(max=2).reshape(-1), reduction='none')
            loss = (loss * valid.reshape(-1)).sum() / valid.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), K.GRAD_CLIP)
            opt.step()
            sched.step()
    net.eval()
    return {'net': net, 'med': med, 'iqr': iqr, 'cols': list(cols), 'seed': int(seed), 'per_epoch': per_epoch}


def predict(model, df, it):
    """(T, 3) probabilità H, A, D: media della previsione diretta e di quella a squadre scambiate (rimesse a posto)."""
    net, med, iqr, cols = model['net'], model['med'], model['iqr'], model['cols']
    dev = next(net.parameters()).device
    T = len(df)
    probs = []
    for swap in (False, True):
        full = np.full((T, 3), 1 / 3, dtype=np.float32)
        for x, _, _, sel in _seqs(df, it, cols, med, iqr, swap=swap):
            with torch.no_grad():
                p = torch.softmax(net(torch.from_numpy(x[None]).to(dev)), -1)[0].cpu().numpy()
            if swap:
                p = p[:, [1, 0, 2]]
            a, b = sel[0], sel[-1] + 1      # da 5 a 25 frame al secondo: interpolazione tra i frame vicini
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
                'cols': list(model['cols']), 'seed': int(model['seed'])}, path)


def load(path):
    """Il modello salvato, sul processore e in modalità previsione."""
    obj = torch.load(path, map_location='cpu', weights_only=True)
    net = Net(obj['n_in'])
    net.load_state_dict(obj['state_dict'])
    net.eval()
    return {'net': net, 'med': obj['med'].numpy(), 'iqr': obj['iqr'].numpy(), 'cols': obj['cols'], 'seed': obj['seed']}
