"""Raccolta centralizzata di tutte le costanti e iperparametri usati in modelli_mv.

Ogni costante è documentata con:
- Cosa rappresenta fisicamente o algoritmicamente;
- Da dove deriva (specifiche ufficiali FIFA EPTS, analisi cinematica del tracking,
  scelte di modellazione, ottimizzazione per coordinate del Viterbi, ecc.);
- I file sorgenti originari in cui era definita.
"""
import numpy as np

# ==================================================================================================
# 1. FORMATO DATI, CAMPO E METRICHE DI GARA (formato_comune.py, dati.py)
# ==================================================================================================

# Frame al secondo del tracking ottico (FIFA EPTS / Metrica Sports).
# 1 frame = 1/25 s = 0.040 s (40 millisecondi).
FPS = 25

# Indici numerici per identificare la squadra di casa (0) e la squadra ospite (1).
HOME = 0
AWAY = 1

# Mappatura standard dei codici di classe del possesso:
# 0 = Casa (Home), 1 = Trasferta (Away), 2 = Palla morta/gioco fermo (Dead), 3 = Fuori tempo di gioco.
H, A, D, X = 0, 1, 2, 3
CODE = {'H': 0, 'A': 1, 'D': 2, 'X': 3}

# Array inverso per convertire l'indice numerico (0..3) nella lettera corrispondente per i CSV di output.
LETTERS = np.array(['H', 'A', 'D', 'X'])

# Numero totale di frame del tracking della partita ufficiale Barça–Athletic (data/tactical/).
# Definito nelle specifiche ufficiali del dataset (games.py e DATA.md).
OUR_FRAMES = 199319

# Dimensioni del campo da gioco di Barça–Athletic in metri (lunghezza X = 105.3 m, larghezza Y = 68.0 m).
# Fornite dai metadati XML della telecamera tattica del Camp Nou.
OUR_LENGTH = 105.3
OUR_WIDTH = 68.0

# Intervalli di frame inclusivi dei due tempi regolamentari di Barça–Athletic (0-based):
# Tempo 1: dal frame 11.185 al frame 90.651
# Tempo 2: dal frame 110.808 al frame 190.197
# Tutti i frame intermedi o fuori da questi estremi hanno period=0 ed etichetta 'X'.
OUR_HALVES = ((1, 11185, 90651), (2, 110808, 190197))

# Tolleranza in metri oltre le linee perimetrali del campo (fondo e laterali).
# Se la palla supera (L/2 + 0.5) o (W/2 + 0.5) viene marcata come fuori campo (palla_fuori = 1).
# I 0.5 m coprono il raggio della palla (~11 cm) e le incertezze del tracking ottico sporcato.
FIELD_BOUND_TOLERANCE = 0.5

# Identificativo convenzionale della partita di gara (fonte 'barca', ID 'BAR-ATH').
BARCA = ('barca', 'BAR-ATH')

# Le tre sorgenti delle 30 partite esterne di allenamento e validazione:
# - 'dfl': 7 partite della Bundesliga (tracking DFL, qualità alta, etichette originali su ogni tocco)
# - 'metrica': 3 partite dimostrative Metrica Sports (qualità alta, tracking sintetico/reale)
# - 'skillcorner': 20 partite con tracking automatico da riprese TV (molto rumore e occlusioni)
SOURCES = ('dfl', 'metrica', 'skillcorner')

# Le 10 partite con tracking tattico ad alta precisione su cui si valuta e si calibra il modello
# (la seconda pagina della dashboard si basa sulla media di queste 10 partite).
DM_SOURCES = ('dfl', 'metrica')
DM = DM_SOURCES

# Elenco completo e fisso delle 30 partite esterne (dati.py).
MATCHES = {
    'dfl': ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9'],
    'metrica': ['Sample_Game_1', 'Sample_Game_2', 'Sample_Game_3'],
    'skillcorner': [
        '1874553', '1886347', '1899585', '1925299', '1927964', '1953632', '1959846', '1986691',
        '1996435', '1996436', '2006229', '2006363', '2007448', '2007721', '2010085', '2011166',
        '2013725', '2015213', '2016236', '2017461',
    ],
}

# Versione della tabella delle caratteristiche salvata in cache.
VERSION = 'v2'


# ==================================================================================================
# 2. CARATTERISTICHE CINEMATICHE E SOGLIE SPAZIALI (caratteristiche.py)
# ==================================================================================================

# Semifinestra (in frame) per il calcolo delle velocità e accelerazioni centrate:
# dt = 2 * SPEED_HALF = 4 frame = 0.16 secondi.
# Ottenuto empiricamente: una differenza a 1 solo frame amplifica il rumore dei sensori;
# una finestra di ±2 frame smussa il jitter senza introdurre ritardo percettibile nei cambi di direzione.
SPEED_HALF = 2

# Soglia di velocità in m/s (2.0 m/s = 7.2 km/h):
# Al di sotto di questa velocità un calciatore è considerato "al passo" o fermo.
# Utilizzata per la feature `quota_cammina` (percentuale di giocatori che camminano),
# indicatore fondamentale per distinguere il gioco fermo dal gioco vivo.
WALK = 2.0
WALK_SPEED_THRESH = 2.0

# Raggio in metri (3.0 m) per individuare i giocatori "vicini" al pallone.
# Utilizzato per contare la pressione immediata (`vicini_casa`, `vicini_ospiti`).
NEAR = 3.0
NEAR_DIST_THRESH = 3.0

# Raggio in metri (1.8 m) per considerare la palla "toccata" o controllata da un giocatore.
# È impostato a 1.8 m (invece di ~1.0 m reale) perché nel dataset sporcato dell'hackathon
# le coordinate del pallone sono affette da rumore e possono essere traslate fino a ~0.8 m.
TOUCH = 1.8
TOUCH_DIST_THRESH = 1.8

# Semifinestre temporali centrate (in frame) per le medie mobili delle feature di base:
# - 12 frame: ±0.48 s (~±0.5 s, dinamica immediata del duello)
# - 37 frame: ±1.48 s (~±1.5 s, possesso a breve termine)
# - 112 frame: ±4.48 s (~±4.5 s, fase di manovra e posizionamento di squadra)
WINDOWS = (12, 37, 112)
CONVOLUTION_HALF_WINDOWS = WINDOWS

# Finestre asimmetriche (in frame) per il calcolo delle medie passate (_p) e future (_f):
# - 25 frame = 1.0 s (inerzia o prospettiva immediata)
# - 75 frame = 3.0 s (contesto a medio termine dell'azione)
ASYMMETRIC_WINDOWS = (25, 75)

# Finestre temporali (in frame) per contare i tocchi ricevuti e la tenuta della palla:
# - 50 frame = 2.0 s
# - 125 frame = 5.0 s
# Consentono al modello di catturare le regole ufficiali ("più di due passaggi di fila", "tiene palla").
TOUCH_COUNT_WINDOWS = (50, 125)

# Saturazione massima (in frame) per il tempo trascorso dall'ultima vista palla/tocco e fino alla prossima.
# 1500 frame = 60.0 secondi. Se la palla non si vede da oltre 1 minuto, la feature satura a 1500.
GAP_CLIP_FRAMES = 1500

# Valore di penalità additivo (in frame, pari a 10.0 s) assegnato a `da_ultima_palla` o `a_prossima_palla`
# quando la palla non è mai stata vista dall'inizio del tempo o fino alla fine del tempo.
NO_BALL_INITIAL_OFFSET = 250

# Limite di clipping per la serie di passaggi consecutivi della stessa squadra (`_streak`).
# Compreso tra -20 (20 passaggi consecutivi ospiti) e +20 (20 passaggi consecutivi casa).
MAX_STREAK = 20

# Percentile calcolato sulle velocità di tutti i giocatori in campo nel singolo frame (90° percentile).
# Rappresenta la velocità di punta dei giocatori più attivi nell'azione, isolando eventuali outlier.
PERCENTILE_VELOCITY = 90


# ==================================================================================================
# 3. CAMPIONAMENTO, ADDESTRAMENTO E CROSS-VALIDATION (esperimento.py, famiglie.py, riproduci.py)
# ==================================================================================================

# Seme casuale globale (Seed 2027, anno dell'hackathon) per garantire la riproducibilità totale
# della divisione dei fold, dell'inizializzazione dei pesi e dei modelli stocastici.
SEED = 2027

# Numero di gruppi (fold) per la Cross-Validation bilanciata per fonte.
# Ogni gruppo contiene 6 partite intere (DFL, Metrica, SkillCorner bilanciate).
N_FOLDS = 5

# Passo di campionamento temporale in fase di addestramento dei modelli tabulari (HGB, XGB, LGBM):
# 1 frame ogni 8 (0.32 s = ~3.1 Hz).
# I frame contigui a 25 fps sono quasi ridondanti; campionare ogni 8 frame riduce il tempo di calcolo
# dell'87% senza perdere accuratezza nelle distribuzioni delle feature.
STEP = 8
TRAIN_SUBSAMPLE_STEP = 8

# Colonne di metadati del DataFrame delle caratteristiche escluse dal training set.
META = ('frame', 'period', 'y')

# Peso relativo assegnato ai frame delle partite SkillCorner in fase di addestramento (1/4 = 0.25).
# Le partite SkillCorner sono 20 ma derivano da riprese televisive con frequenti occlusioni.
# Pesandole 0.25, contano complessivamente quanto 5 partite tattiche, evitando che il rumore televisivo
# sovrasti il segnale pulito delle 10 partite DFL + Metrica.
TV_WEIGHT = 0.25
SOURCE_WEIGHTS = {'dfl': 1.0, 'metrica': 1.0, 'skillcorner': TV_WEIGHT}

# Ponderazione basata sulla durata dei possessi (`short_spell_weights`):
# Assegna un peso w = sqrt(mediana_durata / durata_possesso), limitato nell'intervallo [1/3, 3.0].
# I possessi brevi (1-2 secondi) sono i più difficili da prevedere ed erano spesso ignorati
# dal modello: dare loro peso maggiore aumenta sensibilmente lo spellF1 sulle azioni rapide.
SHORT_SPELL_WEIGHT_MIN = 1.0 / 3.0
SHORT_SPELL_WEIGHT_MAX = 3.0

# Semifinestra temporale (±5 frame = ±0.2 s) per definire l'intorno di un cambio di possesso
# per l'addestramento del modello ausiliario degli eventi (`esperimento.py`).
EV_HALF = 5
EVENT_HALF_WINDOW = 5

# Sottocampionamento dei frame con evento (1 ogni 2) e senza evento (1 ogni 16) con rispettivi pesi
# per bilanciare l'estrema rarità dei cambi di possesso nel dataset.
EVENT_SUBSAMPLE_POS = 2
EVENT_SUBSAMPLE_NEG = 16
EVENT_WEIGHT_POS = 2.0
EVENT_WEIGHT_NEG = 16.0

# Arrotondamento a 4 cifre decimali applicato alle probabilità dell'ensemble prima del Viterbi.
# Essenziale per garantire che l'algoritmo produca esattamente le stesse etichette bit-per-bit
# indipendentemente dall'architettura del processore (x86 vs ARM/Apple Silicon, Windows vs macOS).
DECIMALS = 4
PROB_ROUND_DECIMALS = 4

# Numero di thread per modelli gradient boosting (XGBoost, LightGBM).
THREADS = 8


# ==================================================================================================
# 4. ARCHITETTURA E IPERPARAMETRI DELLA RETE NEURALE (rete.py)
# ==================================================================================================

# Passo di campionamento temporale della sequenza per la rete neurale convoluzionale:
# 25 frame al secondo (passo nativo 0.040 s, STRIDE = 1). La rete analizza tutti i frame
# a 25 Hz senza interpolazione o sfocatura temporale.
STRIDE = 1
NET_STRIDE = 1

# 1500 step a 25 fps = 60 secondi esatti di partita continua.
CROP = 1500
NET_CROP = 1500

# Numero di epoche di addestramento per ciascuna rete neurale (default: 20 epoche).
EPOCHS = 20
NET_EPOCHS = 20

# Canali intermedi della rete neurale convoluzionale 1D.
CHANNELS = 96
NET_CHANNELS = 96

# Dilatazioni temporali dei 10 blocchi convoluzionali residui:
# Dilatazioni: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 passi.
# Campo ricettivo risultante: ±(1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512) = ±1023 passi a 25 fps,
# ovvero esattamente ±40.92 secondi di contesto temporale (passato e futuro) attorno a ciascun frame.
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
NET_DILATIONS = DILATIONS

# Dropout applicato all'interno di ciascun blocco residuo per prevenire l'overfitting.
DROPOUT = 0.15
NET_DROPOUT = DROPOUT

# Learning rate massimo impostato per l'ottimizzatore AdamW con scheduler OneCycleLR.
LR = 2e-3
NET_MAX_LR = LR

# Decadimento del peso (regolarizzazione L2) per AdamW.
WEIGHT_DECAY = 1e-4
NET_WEIGHT_DECAY = WEIGHT_DECAY

# Dimensione del batch di sequenze (32 spezzoni da 60 secondi campionati proporzionalmente alla durata).
BATCH_SIZE = 32
NET_BATCH_SIZE = BATCH_SIZE

# Soglia di clipping della norma del gradiente (grad norm clipping).
GRAD_CLIP = 1.0
NET_GRAD_CLIP = GRAD_CLIP

# Passo di campionamento delle feature (1 riga ogni 40 frame = 1.6 s) per calcolare in modo
# rapido e robusto la mediana e lo scarto interquartile (IQR) di normalizzazione.
NORM_SUBSAMPLE_STEP = 40
NET_NORM_SUBSAMPLE_STEP = NORM_SUBSAMPLE_STEP

# Clipping delle feature standardizzate con (x - med) / IQR per evitare esplosione dei valori dovuta a outlier:
# I valori sono limitati nell'intervallo [-6.0, +6.0].
NET_FEATURE_CLIP = (-6.0, 6.0)

# Numero di reti neurali convoluzionali indipendenti nell'ensemble finale (N_NETS = 3).
# Ciascuna rete ha la stessa architettura ma viene inizializzata con un seme differente
# (seed, seed + 1000, seed + 2000), riducendo la varianza delle predizioni.
N_NETS = 3

# Nome del modello di riferimento
MODEL_NAME = 'mv_hgb_rete3'
NAME = MODEL_NAME


# ==================================================================================================
# 5. IPERPARAMETRI MODELLI AD ALBERI (esperimento.py, famiglie.py)
# ==================================================================================================

# Iperparametri ottimali per HistGradientBoostingClassifier di scikit-learn:
HGB_PARAMS = {
    'max_iter': 500,               # Massimo numero di alberi di boosting
    'learning_rate': 0.08,          # Tasso di apprendimento per ogni albero aggiunto
    'max_leaf_nodes': 63,          # Numero massimo di foglie per albero (~profondità 6)
    'min_samples_leaf': 200,       # Minimo numero di campioni per foglia (previene memorizzazione rumore)
    'l2_regularization': 1.0,       # Penalità L2 sui pesi delle foglie
    'early_stopping': True,        # Attiva l'arresto anticipato se la perdita non scende
    'validation_fraction': 0.1,    # Frazione interna dei dati tenuta per l'early stopping
    'n_iter_no_change': 25,        # Tolleranza: ferma l'addestramento se non migliora per 25 iterazioni
    'random_state': SEED,          # Riproducibilità del boosting
}

# Iperparametri per XGBoost sperimentato in famiglie.py
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'max_depth': 8,
    'eta': 0.06,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 50,
    'reg_lambda': 2.0,
    'eval_metric': 'mlogloss',
    'nthread': THREADS,
    'seed': SEED,
}

# Iperparametri per LightGBM sperimentato in famiglie.py
LGBM_PARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 2.0,
    'num_threads': THREADS,
    'seed': SEED,
    'verbose': -1,
}


# ==================================================================================================
# 6. DECODIFICA DI VITERBI E PARAMETRI CONGELATI (modello.py, congelato/parametri_viterbi.json)
# ==================================================================================================

# Valore di infinito per la matrice di programmazione dinamica di Viterbi.
INF = 1e18
INF_COST = INF

# Valore minimo di probabilità ammesso per la conversione in costi (-log(p)):
# Evita divisioni per zero o costi infiniti (p=0 diventa -log(1e-4) ~ 9.21).
PROB_CLIP_MIN = 1e-4

# Valore minimo di probabilità ammesso per le probabilità di eventi ausiliari.
EVENT_PROB_CLIP_MIN = 1e-3

# Parametri di default prima della taratura
DEFAULT = dict(turn=6.0, stop=6.0, rest=6.0, m_pos=20, m_dead=40, bias_d=0.0)
DEFAULT_VITERBI_PARAMS = DEFAULT

# Punto iniziale della ricerca per coordinate, non i parametri del congelato.
START = dict(turn=9.0, stop=13.0, rest=9.0, m_pos=5, m_dead=60, bias_d=-0.5)
VITERBI_SEARCH_START = START
# Copia documentale dei parametri congelati. L'inferenza carica sempre il JSON
# dell'artefatto; non usare questo riferimento per calibrare una nuova prova.
VITERBI_PARAMS_FROZEN = dict(turn=9.0, stop=4.0, rest=9.0, m_pos=5, m_dead=60, bias_d=-0.5)

# Griglia di ricerca per l'ottimizzazione per coordinate del decodificatore di Viterbi (esperimento.py).
GRID = {
    'turn': [2, 4, 6, 9, 13, 18, 25, 35],
    'stop': [2, 4, 6, 9, 13, 18, 25],
    'rest': [2, 4, 6, 9, 13, 18, 25],
    'm_pos': [5, 10, 15, 20, 30, 40, 55, 75],
    'm_dead': [10, 20, 30, 40, 60, 85],
    'bias_d': [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
}
VITERBI_GRID = GRID


# ==================================================================================================
# 7. FILTRO DEI RIMPALLI SULLE ETICHETTE DFL (etichette.py)
# ==================================================================================================

# Raggio in metri (1.60 m) entro cui un giocatore può raggiungere e controllare fisicamente la palla.
REACH = 1.60
RICOCHET_REACH = REACH

# Velocità massima della palla in m/s (16.0 m/s = 57.6 km/h) per un passaggio controllato.
# Al di sopra di questa velocità si tratta di un tiro o di una respinta violenta, non di controllo.
PASS_SPEED = 16.0
RICOCHET_PASS_SPEED = PASS_SPEED

# Durata minima in secondi (0.24 s = 6 frame a 25 fps) di controllo continuo
# per considerare un contatto come vero possesso intenzionale e non come semplice rimpallo.
MIN_CTRL_S = 0.24
RICOCHET_MIN_CTRL_S = MIN_CTRL_S

# Durata massima in secondi (1.60 s = 40 frame a 25 fps) per considerare un segmento
# come potenziale possesso isolato/breve da sottoporre al filtro dei rimpalli.
MAX_PROBE_S = 1.60
RICOCHET_MAX_PROBE_S = MAX_PROBE_S
