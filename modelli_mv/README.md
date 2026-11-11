# modelli_mv — modello `mv_hgb_rete3`

È il nostro modello migliore: un **ensemble** che per ogni frame dice H, A, D o X, in tre passi:

1. **HGB**: gradient boosting di scikit-learn, frame per frame, su 142 caratteristiche del tracking.
2. **Rete neurale convoluzionale**: guarda la sequenza a 5 frame al secondo, con ±25 s di contesto. Tre semi diversi.
3. **Media delle 4 probabilità** (HGB + 3 reti), poi **Viterbi** con durate minime.

Si allena sulle 30 partite esterne sporcate (7 DFL, 3 Metrica, 20 SkillCorner), con le etichette delle versioni pulite. Le SkillCorner pesano 1/4. Il Barça non entra mai.

**Il riferimento è congelato in `congelato/`.** Ci sono i pesi di tutti i modelli, i parametri del Viterbi e le impronte dei risultati. Con le versioni giuste e gli stessi dati, ognuno sul suo computer ottiene **esattamente** gli stessi numeri:
- il CSV del Barça;
- la prima pagina della dashboard (5 minuti);
- la seconda pagina (10 partite DFL + Metrica).

## 1. Installazione (una volta sola)

Dalla cartella della repo: su Mac nel Terminale, su Windows nell'Anaconda Prompt.

```bash
conda env create -f modelli_mv/environment.yml
conda activate mtah
```

Così si installano Python 3.11 e le versioni esatte delle librerie, da conda-forge. Va bene su Mac, Windows e Linux.

In alternativa, solo su Windows e Linux, si può usare pip: `pip install -r modelli_mv/requirements.txt`. Su un Mac con Anaconda Intel pip non trova torch 2.14.1.

## 2. Dati

I dati non stanno su GitHub: vanno in `data/`.

| servono per | file |
|---|---|
| `prevedi` | `data/tactical/BAR-ATH_tactical_atd.txt` e `BAR-ATH_tactical_atd.xml` (quelli della gara) |
| `esterne`, `allena` | per ognuna delle 30 partite: `data/altre_partite/<fonte>_sporcate/<id>/<id>_atd.txt` e `_atd.xml`, e `data/altre_partite/<fonte>_pulite/<id>/<id>_possession_gt.csv` (dal Drive di Matteo) |

L'elenco delle 30 partite è in `dati.py` (`MATCHES`). Altre cartelle dentro `data/altre_partite/` vengono ignorate. `controlla` dice se i tuoi file sono identici a quelli usati per il riferimento.

## 3. Ripartire dal riferimento

```bash
python modelli_mv/riproduci.py controlla    # versioni e dati: deve finire con "Tutto in ordine."
python modelli_mv/riproduci.py prevedi      # CSV del Barça: evaluation/results/mv_hgb_rete3.csv
python modelli_mv/riproduci.py esterne      # 10 partite: data/risultati_esterne/mv_hgb_rete3/
python dashboard/app.py                     # la dashboard, con mv_hgb_rete3 su tutte e due le pagine
```

`prevedi` ed `esterne` devono finire tutti e due con:

```
RISULTATO: IDENTICO al riferimento (stesse impronte SHA-256).
```

Numeri del riferimento:

| dove | spellF1 | altro |
|---|---|---|
| 10 partite DFL + Metrica (seconda pagina) | **0,7406** | DFL 0,704 · Metrica 0,826 · partita peggiore 0,644 |
| 5 minuti pubblici (prima pagina) | **0,7636** | macroF1 0,9434 |
| CSV del Barça | — | 321 possessi |

I 5 minuti pubblici servono solo a controllare il file del Barça. Il modello si sceglie sulle 10 partite.

## 4. Provare a migliorarlo

1. **Cambi il codice**: per esempio `caratteristiche.py`, `rete.py`, oppure i pesi in `esperimento.py`.
2. **Rifai tutta la ricetta** con un nome tuo:

   ```bash
   python modelli_mv/riproduci.py allena --nome ste_prova1
   ```

   I passi sono:
   - validazione a 5 gruppi: ogni partita è prevista da modelli che non l'hanno vista;
   - parametri del Viterbi, regolati sulle 10 partite;
   - modello finale per il Barça.

   Ci vuole circa 1 ora su un Mac con scheda grafica, di più sul processore.
3. **Confronta.** Lo script stampa i numeri accanto a quelli del riferimento. Nella dashboard `ste_prova1` compare su tutte e due le pagine, accanto a `mv_hgb_rete3`. I pesi vanno in `data/cache_mv/modelli/ste_prova1/`.

**Attenzione al rumore.** L'allenamento della rete su scheda grafica non è deterministico. La stessa ricetta riallenata dà numeri leggermente diversi, con circa ±0,003–0,005 di spellF1 medio sulle 10 partite: il 10 novembre la stessa ricetta aveva dato 0,7436, oggi 0,7406. Un miglioramento più piccolo di così non si distingue dal caso.

Per conoscere il tuo rumore, la prima volta lancia `allena --nome base` **senza cambiare niente**.

Quando una prova batte chiaramente il riferimento, la si congela al suo posto. Lo fa Matteo, con `allena --congela`, e poi la pubblica.

## 5. Comandi

| comando | cosa fa | tempo |
|---|---|---|
| `controlla [--solo-barca]` | versioni delle librerie e impronte dei 92 file dei dati (`--solo-barca`: solo i 2 della gara) | meno di 1 min |
| `prevedi` | CSV del Barça dai pesi congelati, validato e confrontato col riferimento | meno di 1 min |
| `esterne` | risposte sulle 10 partite dai pesi congelati dei 5 gruppi, punteggi, confronto col riferimento | circa 5 min (2 in più la prima volta, per la cache) |
| `allena --nome NOME` | tutta la ricetta da capo con il codice attuale | circa 1 ora su Mac con scheda grafica |

## 6. Problemi

- **`controlla` dice `DIVERSO` o `MANCA` su un file dei dati**: la tua copia non è quella del riferimento. Riscaricala dal Drive. Gli a capo Windows non contano.
- **`controlla` dice `<-- serve x.y.z`**: rifai l'ambiente da zero con `conda env remove -n mtah`, poi `conda env create -f modelli_mv/environment.yml`.
- **`prevedi` o `esterne` non dicono IDENTICO, anche se `controlla` è in ordine**: manda a Matteo l'output completo dei tre comandi.
- **Hai codice o cache delle versioni precedenti**: non serve cancellare niente. La cache delle caratteristiche (`data/cache_mv/v2_<impronta>/`) ha nel nome l'impronta del codice, quindi si rifà da sola quando il codice cambia.
- **Allenare su Windows è lento**: senza una scheda NVIDIA con torch CUDA la rete si allena sul processore. Per `prevedi` ed `esterne` non serve. Si può imporre il dispositivo con la variabile `MV_DEVICE=cpu|cuda|mps`.

## 7. File

| file | cosa fa |
|---|---|
| `riproduci.py` | i comandi qui sopra |
| `congelato/` | il riferimento: `finale/` (modello per il Barça), `gruppi/gruppo0-4/` (modelli della validazione), `parametri_viterbi.json`, impronte attese dei risultati (`attesi*.json`) e dei dati (`dati_attesi.json`) |
| `formato_comune.py` | legge i file di tracking (FIFA EPTS) |
| `caratteristiche.py` | le 142 caratteristiche per frame (palla, squadre, tocchi, finestre nel tempo) e lo scambio delle squadre |
| `dati.py` | le 30 partite e la cache delle caratteristiche in `data/cache_mv/` |
| `esperimento.py` | righe di allenamento, pesi, HGB, gruppi di partite, regolazione del Viterbi |
| `rete.py` | la rete neurale: allenamento, previsione, salvataggio |
| `modello.py` | il Viterbi e lo spellF1 veloce (uguale a quello ufficiale) |
| `famiglie.py`, `scegli.py` | le prove del 10 novembre (XGBoost, LightGBM, cascata, rete) e il confronto fra modelli |
| `prevedi.py`, `confronto_dati.py`, `fonte_fuori.py`, `etichette.py` | prove precedenti (quali dati usare, SkillCorner, etichette senza rimpalli) |

Nella repo vanno solo codice, pesi e impronte. Dati, cache e CSV restano fuori: `data/` ed `evaluation/results/*.csv` sono già in `.gitignore`.
