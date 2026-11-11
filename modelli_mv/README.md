# modelli_mv — ottenere esattamente i nostri risultati e partire da lì

Questa cartella contiene il nostro modello migliore, **`mv_hgb_rete3`**, congelato. Chi segue questa guida ottiene sul suo computer **esattamente** gli stessi risultati, frame per frame:

| cosa | dove si vede | spellF1 | altro |
|---|---|---|---|
| 10 partite esterne DFL + Metrica | dashboard, seconda pagina | **0,7406** (nella dashboard: 74,1) | macroF1 0,9376 · DFL 0,704 · Metrica 0,826 · partita peggiore 0,644 |
| 5 minuti pubblici del Barça | dashboard, prima pagina | **0,7636** (nella dashboard: 76,4) | macroF1 0,9434 (94,3) |
| CSV del Barça | `evaluation/results/mv_hgb_rete3.csv` | — | 321 possessi; H 49.972 · A 32.337 · D 76.548 · X 40.462 frame |

"Esattamente" vuol dire che lo script confronta l'impronta SHA-256 di ogni file prodotto con quella del riferimento e scrive `IDENTICO`.

Il modello si giudica e si sceglie sulle **10 partite esterne**. I 5 minuti pubblici servono solo a controllare il file del Barça: le regole vietano di usarli per regolare il modello.

In breve, i passi sono:

```bash
git clone https://github.com/SAM-Analytics-2027/more-than-a-hack-2027.git mtah-mv
cd mtah-mv
git switch MV
conda env create -f modelli_mv/environment.yml
conda activate mtah
#   ...mettere i dati in data/ (passo 3)...
python modelli_mv/riproduci.py controlla
python modelli_mv/riproduci.py prevedi
python modelli_mv/riproduci.py esterne
python dashboard/app.py
```

Sotto, ogni passo nel dettaglio.

---

## Passo 0 — Cosa serve

- **git** e **conda**: Anaconda o Miniconda, va bene qualunque versione recente.
- **Almeno 10 GB liberi**: circa 3,5 GB di dati, 1,6 GB di cache, circa 2 GB per l'ambiente Python.
- **I file della gara** (Barça–Athletic), dal download dell'hackathon.
- **Le 30 partite esterne**, dal Drive di Matteo. Servono solo per `esterne` e `allena`; per `prevedi` no.

Su **Windows** tutti i comandi vanno scritti nell'**Anaconda Prompt** (Start → "Anaconda Prompt"), non nel Prompt dei comandi normale. I percorsi con `/` funzionano anche su Windows.

## Passo 1 — Il codice

Consigliato: una **cartella nuova**, così non si mescola con modifiche fatte in passato.

```bash
git clone https://github.com/SAM-Analytics-2027/more-than-a-hack-2027.git mtah-mv
cd mtah-mv
git switch MV
```

Se usi la cartella che hai già, prima controlla di non avere modifiche locali:

```bash
git status                 # deve dire "nothing to commit, working tree clean"
git fetch origin
git switch MV
git pull
```

Se `git status` mostra file modificati, mettili da parte con `git stash`, oppure usa la cartella nuova.

## Passo 2 — L'ambiente Python

```bash
conda env create -f modelli_mv/environment.yml
conda activate mtah
```

Ci vogliono 5–15 minuti. L'ambiente si chiama `mtah`: Python 3.11 con le versioni esatte di numpy, pandas, pyarrow, scikit-learn, scipy, joblib, numba e torch, tutte da conda-forge. Va bene su Mac (Intel o Apple), Windows e Linux.

- **Ogni volta** che apri un terminale nuovo, scrivi prima `conda activate mtah`.
- Se esiste già un ambiente `mtah`, scrivi prima `conda env remove -n mtah`.
- Senza conda, solo su Windows e Linux: `pip install -r modelli_mv/requirements.txt` in un Python 3.11. Su un Mac con Anaconda Intel pip non trova torch 2.14.1: lì usa conda.

**Perché le versioni esatte.** Con versioni diverse l'ultima cifra di alcuni calcoli può cambiare, e con lei qualche frame. `controlla` verifica le versioni.

## Passo 3 — I dati

I dati non stanno su GitHub (`data/` è in `.gitignore`). Vanno messi **dentro la cartella `data/` della repo**, con esattamente questi percorsi:

```
data/
├── tactical/                                  ← dal download dell'hackathon
│   ├── BAR-ATH_tactical_atd.txt
│   └── BAR-ATH_tactical_atd.xml
└── altre_partite/                             ← dal Drive di Matteo
    ├── dfl_sporcate/<id>/<id>_atd.txt         l'input del modello (tracking sporcato)
    ├── dfl_sporcate/<id>/<id>_atd.xml
    ├── dfl_pulite/<id>/<id>_possession_gt.csv  la risposta giusta (solo per allenare e giudicare)
    ├── metrica_sporcate/...   metrica_pulite/...      (stessa struttura)
    └── skillcorner_sporcate/... skillcorner_pulite/... (stessa struttura)
```

Le 30 partite, cioè i nomi delle cartelle `<id>`:

| fonte | id |
|---|---|
| DFL (7) | J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 |
| Metrica (3) | Sample_Game_1, Sample_Game_2, Sample_Game_3 |
| SkillCorner (20) | 1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461 |

In tutto sono 92 file. Delle cartelle `_pulite` serve solo il file `_possession_gt.csv`: gli altri file possono esserci, ma non vengono letti. Altre cartelle dentro `data/altre_partite/` (per esempio copie con più rumore) vengono ignorate, perché l'elenco delle partite è fisso in `dati.py`.

## Passo 4 — Controllo

```bash
python modelli_mv/riproduci.py controlla
```

Deve finire così:

```
2. File dei dati (impronte in modelli_mv/congelato/dati_attesi.json)
   92 file controllati: 92 uguali, 0 mancanti, 0 diversi
Tutto in ordine.
```

Se hai solo i file della gara, usa `controlla --solo-barca`: deve dire `2 file controllati: 2 uguali`.

## Passo 5 — Riprodurre i risultati

```bash
python modelli_mv/riproduci.py prevedi
```

In meno di 1 minuto scrive `evaluation/results/mv_hgb_rete3.csv`. Deve stampare:

```
Barça: scritto evaluation/results/mv_hgb_rete3.csv (15 s), formato valido
   possessi 321  conteggi {'H': 49972, 'A': 32337, 'D': 76548, 'X': 40462}
   5 minuti pubblici (prima pagina della dashboard): spellF1 0.7636  macroF1 0.9434
   RISULTATO: IDENTICO al riferimento (stesse impronte SHA-256).
```

```bash
python modelli_mv/riproduci.py esterne
```

In 3–5 minuti scrive le risposte per le 10 partite in `data/risultati_esterne/mv_hgb_rete3/`. La prima volta servono 2 minuti in più per preparare la cache delle caratteristiche. Deve stampare:

```
Partite esterne: 10 risposte in data/risultati_esterne/mv_hgb_rete3 (150 s)
   spellF1 medio sulle 10 DFL + Metrica: 0.7406  (DFL 0.704, Metrica 0.826, peggiore 0.6443)  | SkillCorner 0.7772
   RISULTATO: IDENTICO al riferimento (stesse impronte SHA-256).
```

I secondi tra parentesi cambiano da computer a computer; tutto il resto deve essere uguale.

**Come funziona.** Non si riallena niente: si usano i pesi salvati in `modelli_mv/congelato/`. Ogni partita esterna è prevista dai modelli del suo gruppo di validazione, cioè da modelli che **non l'hanno mai vista**. Le probabilità sono arrotondate a 4 decimali prima del Viterbi, così lo stesso calcolo dà lo stesso risultato anche su processori diversi.

## Passo 6 — La dashboard

```bash
python dashboard/app.py
```

Si apre il browser su http://127.0.0.1:8765. Se la porta è occupata, usa `python dashboard/app.py --port 8766`.

- **Prima pagina (5 minuti):** `mv_hgb_rete3` con Voto Azioni (spellF1) **76,4** e Voto Frame (macroF1) **94,3**.
- **Seconda pagina** (pulsante "Partite esterne", oppure http://127.0.0.1:8765/esterne): `mv_hgb_rete3` con spellF1 medio **74,1**, DFL 70,4, Metrica 82,6.

La dashboard legge `evaluation/results/*.csv` e `data/risultati_esterne/*/`. Ogni prova che fai compare accanto al riferimento.

---

## Com'è fatto il modello

Per ogni frame dice H, A, D o X.

1. **142 caratteristiche per frame** (`caratteristiche.py`), calcolate dal tracking sporcato:
   - la palla;
   - le due squadre;
   - i giocatori vicini alla palla;
   - i tocchi (chi ha toccato per ultimo, chi tocca dopo, i tocchi di fila);
   - le medie su finestre di tempo;
   - `fonte_tv` (1 per SkillCorner, 0 per tutto il resto, Barça compreso).

   Il campo è girato in modo che la squadra di casa attacchi sempre verso destra.
2. **Quattro modelli di probabilità** H/A/D:
   - un **HGB** (gradient boosting di scikit-learn, `esperimento.make_model`);
   - **3 reti neurali convoluzionali** (`rete.py`), che guardano la sequenza a 5 frame al secondo con ±25 s di contesto, e differiscono solo per il seme.

   Ognuno prevede due volte, con le squadre normali e scambiate, e fa la media.
3. **Media delle 4 probabilità**, arrotondata a 4 decimali.
4. **Viterbi** con durate minime (`modello.py`). Parametri: possesso minimo 5 frame, palla morta minima 60 frame, costi dei cambi 9 (H↔A), 4 (gioco→D) e 9 (D→gioco), spinta verso D −0,5.

**La ricetta di allenamento** (`riproduci.py allena`):
- **Dati:** le 30 partite esterne. L'input è il tracking sporcato, le etichette vengono dalle versioni pulite. I frame SkillCorner pesano 1/4; i possessi brevi pesano di più. Ogni partita entra anche con le squadre scambiate.
- **Validazione:** 5 gruppi di partite intere, bilanciati per fonte (seme 2027). Per ogni gruppo si allenano i 4 modelli sulle altre partite e si prevede il gruppo. Sono i pesi in `congelato/gruppi/`.
- **Viterbi:** i parametri si regolano sulle previsioni delle 10 partite DFL + Metrica: ricerca per coordinate, 2 giri, partendo da `START`. Il punteggio "annidato", con i parametri regolati senza il gruppo giudicato, è 0,7403.
- **Modello finale:** gli stessi 4 modelli allenati su tutte e 30 le partite. Sono i pesi in `congelato/finale/`, e prevedono il Barça.
- **Il Barça non entra mai** né nell'allenamento né nella regolazione.

## Partire da qui per migliorarlo

### 1. La tua prima prova: misura il rumore

```bash
python modelli_mv/riproduci.py allena --nome base_ste
```

Rifà tutta la ricetta **senza cambiare niente**. Ci vuole circa 50 minuti su un Mac con scheda grafica, molto di più sul processore (vedi "Problemi").

Lo script stampa i numeri accanto a quelli del riferimento. I numeri escono **un po' diversi**, perché l'allenamento della rete su scheda grafica non è deterministico. Quella differenza è il rumore: con la stessa ricetta il 10 novembre era uscito 0,7436 sulle 10 partite, oggi 0,7406. **Un miglioramento più piccolo di circa 0,005 non si distingue dal caso.**

### 2. Le tue modifiche

Cambia il codice, poi lancia `allena` con un nome nuovo. Usa le tue iniziali e una parola, per esempio `ste_reteveloce` o `arm_tocchi`.

| se vuoi cambiare | dove |
|---|---|
| peso di SkillCorner, numero e semi delle reti, punto di partenza del Viterbi | in cima a `riproduci.py` (`TV_WEIGHT`, `N_NETS`, `SEED`, `START`) |
| come si combinano i modelli (pesi della media, altri modelli) | `riproduci.py`: `fit_components`, `load_components`, `ensemble_proba` |
| le caratteristiche | `caratteristiche.py`. La cache si rifà da sola, perché il suo nome contiene un'impronta del codice |
| la rete (strati, epoche, risoluzione, contesto) | `rete.py` |
| l'HGB | `esperimento.make_model` |
| i pesi dei possessi brevi | `esperimento.short_spell_weights` |
| il Viterbi e la griglia della regolazione | `modello.py`, `esperimento.GRID` |

### 3. Confronta

- **Nel terminale:** alla fine di `allena` compaiono le righe `confronto con il riferimento mv_hgb_rete3`.
- **Nella dashboard:** la tua prova compare con il suo nome su tutte e due le pagine. Aggiorna il browser.
- **Su disco:** i pesi e le impronte della prova vanno in `data/cache_mv/modelli/<nome>/`, i risultati in `evaluation/results/<nome>.csv` e `data/risultati_esterne/<nome>/`.

### 4. Condividi una prova, che gli altri rifanno identica

1. Il codice va su un tuo branch: `git add`, `git commit`, `git push`.
2. La cartella dei pesi `data/cache_mv/modelli/<nome>/` (circa 46 MB) va sul Drive.
3. Gli altri fanno `git switch` sul tuo branch e mettono la cartella nello stesso percorso. Poi:

   ```bash
   python modelli_mv/riproduci.py prevedi --modello <nome>
   python modelli_mv/riproduci.py esterne --modello <nome>
   ```

   Deve uscire `RISULTATO: IDENTICO a quello atteso per questi pesi`, più il confronto con il riferimento.

### 5. Nuovo riferimento

Quando una prova batte chiaramente il riferimento, sulle 10 partite e sopra il rumore, la si congela al suo posto in `modelli_mv/congelato/`. Lo fa Matteo, con `allena --congela`, e poi la pubblica. Così tutti ripartono di nuovo dallo stesso punto.

---

## Cosa va su GitHub e cosa no

- **Sì:** il codice in `modelli_mv/` e `modelli_mv/congelato/` (pesi, parametri e impronte, 46 MB, nessun file sopra 5 MB).
- **No:** `data/` (dati, cache, pesi delle prove, risultati esterni) ed `evaluation/results/*.csv`, il CSV del Barça. Sono già esclusi da `.gitignore`. Il CSV del Barça non va mai pubblicato.

## Problemi

| messaggio | cosa fare |
|---|---|
| `conda` non è riconosciuto (Windows) | usa l'**Anaconda Prompt**, non il Prompt dei comandi |
| `ModuleNotFoundError` | l'ambiente non è attivo: `conda activate mtah` |
| `controlla`: `<-- serve x.y.z` | versione sbagliata: `conda env remove -n mtah`, poi rifai il passo 2 |
| `controlla`: `MANCA data/...` | il file non c'è o è nel percorso sbagliato: confronta con l'albero del passo 3 |
| `controlla`: `DIVERSO data/...` | la tua copia non è quella del riferimento: riscaricala dal Drive. Gli a capo Windows non contano |
| `prevedi` o `esterne`: `diverso da quello atteso`, ma `controlla` dice "Tutto in ordine." | manda a Matteo l'output completo dei tre comandi |
| `non trovo i pesi di "<nome>"` | la cartella `data/cache_mv/modelli/<nome>/` non c'è, oppure il nome è scritto diverso |
| `allena` lentissimo | senza scheda grafica la rete si allena sul processore e ci vogliono ore. Su Windows con una scheda NVIDIA installa torch con CUDA: la rete la usa da sola. Si può imporre il dispositivo con `MV_DEVICE=cpu`, `cuda` o `mps`. Per `prevedi` ed `esterne` non serve |
| `allena --nome mv_hgb_rete3` rifiutato | è il nome del riferimento: scegline un altro |
| cache o file di versioni vecchie | non serve cancellare niente: la cache `data/cache_mv/v2_<impronta>/` si rifà da sola se cambia il codice |

## File

| file | cosa fa |
|---|---|
| `riproduci.py` | i comandi `controlla`, `prevedi`, `esterne`, `allena`, e la ricetta del modello |
| `congelato/` | il riferimento: `finale/` (modello per il Barça), `gruppi/gruppo0-4/` (modelli della validazione), `parametri_viterbi.json`, `attesi.json` e `attesi_esterne.json` (risultati attesi con le loro impronte), `dati_attesi.json` (impronte dei 92 file di dati) |
| `environment.yml`, `requirements.txt` | le versioni esatte delle librerie |
| `formato_comune.py` | legge i file di tracking (FIFA EPTS) |
| `caratteristiche.py` | le 142 caratteristiche per frame e lo scambio delle squadre |
| `dati.py` | le 30 partite e la cache delle caratteristiche in `data/cache_mv/` |
| `esperimento.py` | righe di allenamento, pesi, HGB, gruppi di partite, regolazione del Viterbi |
| `rete.py` | la rete neurale: allenamento, previsione, salvataggio |
| `modello.py` | il Viterbi e lo spellF1 veloce (dà gli stessi numeri del codice ufficiale) |
| `famiglie.py`, `scegli.py` | le prove del 10 novembre (XGBoost, LightGBM, cascata, rete) e il confronto fra modelli |
| `prevedi.py`, `confronto_dati.py`, `fonte_fuori.py`, `etichette.py` | prove precedenti (quali dati usare, SkillCorner, etichette senza rimpalli): non servono per il riferimento |
