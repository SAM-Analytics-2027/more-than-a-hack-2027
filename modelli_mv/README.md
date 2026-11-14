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
- Senza conda, solo su Windows e Linux: `pip install -r requirements.txt` in un Python 3.11. Su un Mac con Anaconda Intel pip non trova torch 2.14.1: lì usa conda.

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

## Confronto dell'augmentation su partite esterne

**Stato di `aug_sporcate_v2`: pipeline e CSV verificati tecnicamente;
beneficio predittivo non dimostrato.** La procedura qui descritta riproduce
l'esperimento già eseguito. La revisione metodologica e il nuovo confronto
`aug_pulite_v3` sono documentati in [`changelog.md`](../changelog.md).

La nuova implementazione parte dalle **pulite esterne** per generare input
sporcati, stimando il profilo esclusivamente nelle partite del training effettivo.
Confronta base e augmentation con tre repliche complete e calibrazione annidata;
mantiene fisse le dieci sporcate valutate dalla dashboard. Le prove sulle singole
componenti sono diagnostiche e usano il decoder della base. Il nuovo esperimento
è completato: risultati, variabilità e limiti sono riportati nel changelog.
La media delle tre repliche sulle dieci partite è **73,89 spellF1 / 93,89 macroF1**
per la risporcatura, contro **74,00 / 93,83** della base. Non emerge un beneficio
predittivo dimostrabile; restano discrepanze nelle durate delle occlusioni.
La pipeline è verificata tecnicamente, ma questa ricetta non sostituisce il
riferimento congelato. Sono stati verificati 2.790 input aumentati, 120 CSV
esterni e due CSV finali riproducibili; dettagli e limiti nel changelog.
Le sole cache Parquet delle viste di training di questo studio occupano circa
146 GiB; per ripeterlo servono inoltre spazio per dati sorgente, pesi e probabilità.

```powershell
conda run --no-capture-output -n mtah python modelli_mv/studio_augmentation.py --nome aug_pulite_v3 --fase tutto
conda run --no-capture-output -n mtah python modelli_mv/valida_studio_sporcature.py --esperimento aug_pulite_v3 --ricomputa --finale --diagnostica
conda run --no-capture-output -n mtah python modelli_mv/valida_csv_studio.py --esperimento aug_pulite_v3
```

I risultati della nuova prova sono in `data/cache_mv/augmentation_runs/aug_pulite_v3/`;
il log iniziale è `data/cache_mv/aug_pulite_v3.log`. Nella prosecuzione del 14 novembre
le due code indipendenti scrivono in `data/cache_mv/aug_pulite_v3_aug_remaining.log`
e `data/cache_mv/aug_pulite_v3_base_ablations.log`; la finalizzazione scrive in
`data/cache_mv/finalize_after_workers_v3.log`. La verifica dei CSV richiede il confronto
completo: ricostruisce tutte le 120 esportazioni dalle probabilità fuori gruppo
salvate e dai rispettivi decoder, confrontando frame, etichette e byte. Non legge
la ground truth del Barça e non modifica le predizioni.
I CSV esterni compaiono nella pagina `/esterne` con prefisso `aug_pulite_v3__`;
le sei prove sulle componenti hanno inoltre il prefisso `ab_`. I due CSV finali
sono `aug_pulite_v3__base_r0.csv` e `aug_pulite_v3__risporcata_r0.csv` nella cartella
`evaluation/results/` e compaiono nella prima pagina della dashboard quando pronti.
Le repliche sono esperimenti distinti: la loro media non è un ulteriore ensemble.

I comandi seguenti descrivono invece
la precedente prova `aug_sporcate_v2` e richiedono i suoi sorgenti originali per
superare i controlli delle impronte.

Attiva l'ambiente `mtah` prima di avviare il comando, così Windows trova anche le librerie CUDA dell'ambiente.

```bash
conda activate mtah
python modelli_mv/prova_augmentation.py --nome aug_sporcate_v2
```

Il confronto legge **esclusivamente gli ATD sporcati** in
`data/altre_partite/<fonte>_sporcate/<id>/`. Dalle cartelle `_pulite` legge soltanto
`<id>_possession_gt.csv`, mai le coordinate pulite. Questo vale per tutte le 30 partite.
È una scelta di questo esperimento, non una restrizione del regolamento:
si possono usare le coordinate pulite esterne per generare nuovi input sporcati.
La vista chiamata `originale` è quindi **la sporcata di partenza**, senza ulteriori perturbazioni.
Per ogni partita calcola le feature su questa vista, su una vista con buchi contigui della palla e
di singoli giocatori e su una vista con errore di posizione della palla correlato nel tempo. Durata dei
buchi ed entità dell'errore dipendono dalle osservazioni nel tracking esterno. Una quarta vista combina
le perturbazioni con un seme distinto e serve allo stress test. L'augmentation preserva etichette,
frame, tempi e squadre. Barça–Athletic e i cinque minuti pubblici restano fuori dalla procedura.
La corrispondenza delle perturbazioni con una distribuzione di errori realistica
non è stata dimostrata; aggiungerle al tracking già sporcato può alterare la
distribuzione del training rispetto al benchmark.

Le quattro ricette confrontano la base, le perturbazioni del tracking, il campionamento del 25% dei crop
della rete intorno alle transizioni e la combinazione. Quando sono attive le perturbazioni, ogni riga
HGB o crop della rete sceglie una delle tre viste di training con probabilità 50%, 25% e 25%. Una vista
sostituisce quella originale nello stesso budget di campionamento. I pesi delle fonti sono definiti in
`constants.SOURCE_WEIGHTS` e valgono per tutti i modelli del confronto.

Per ciascuno dei cinque fold, il gruppo successivo è riservato all'early stopping di HGB e alla taratura
del Viterbi. **Ogni ricetta calibra il proprio Viterbi**, soltanto sulla vista sporcata di partenza
delle DFL e Metrica di calibrazione. Training, calibrazione e test non condividono partite.
Questo lascia in media 18 partite per il training e soltanto 1–3 DFL/Metrica
per calibrare ciascun decoder, con variabilità elevata dei parametri scelti.
Il confronto non isola le cause dello scarto rispetto al congelato, allenato
su una media di 24 partite per fold: include anche modifiche a feature e rete.
L'obiettivo è la media per partita DFL + Metrica, con soglia macroF1 di 0,90, e poi spellF1.
La vista stressata è diagnostica e non determina parametri o vincitore. Il test comprende
la vista originale e quella stressata di ogni partita fuori dal training e dalla calibrazione.
Il riepilogo riporta lo SpellF1 medio per partita, il macroF1 e, a IoU 0,50, la precisione su tutti i
possessi e il recall su quelli sotto due secondi. I CSV fuori gruppo vengono esportati automaticamente
in `data/risultati_esterne/<nome>__<ricetta>/` e appaiono nella dashboard `/esterne`.
Il riferimento congelato proviene da un altro protocollo: per misurare l'effetto dell'augmentation
confrontare soprattutto le quattro ricette di **questo stesso esperimento**.
La selezione della ricetta usa questa cross-validation di sviluppo; i punteggi non sono una stima
su un ulteriore test indipendente dalla selezione.

I risultati e i pesi dei fold sono in `data/cache_mv/augmentation_runs/<nome>/`. Ripetere il comando con
lo stesso nome e la stessa configurazione riprende dai pesi o dai fold già salvati. Input, codice,
versioni e artefatti completati sono identificati da impronte SHA-256. Le vecchie cache, costruite
sul tracking pulito, non vengono riusate. Un cambiamento di configurazione richiede un nuovo nome.
Per un controllo rapido del percorso completo si può usare
`--nome aug_sporcate_smoke_v2 --fold 0 --n-reti 1 --epoche 1 --smoke`.
Gli esperimenti smoke non possono generare il modello finale.

I nuovi training includono il contesto dei crop senza assegnargli loss e tengono separate le
interruzioni osservabili del tracking; tutte le ricette usano gli stessi controlli e la stessa rete.
I checkpoint precedenti mantengono la modalità di inferenza e le feature legacy.

Per verificare dati, cache e risultati:

```bash
python -m unittest discover -s tests -v
python modelli_mv/valida_augmentation.py --ricomputa
python modelli_mv/valida_risultati_augmentation.py --esperimento aug_sporcate_v2
```

`valida_augmentation.py` controlla le 120 viste delle 30 partite, le etichette,
la provenienza e l'assenza di Inf; `--ricomputa` richiede anche uguaglianza esatta
con le feature ricostruite dagli ATD sporcati. Il report è
`data/cache_mv/validazione_augmentation.json`.
`valida_risultati_augmentation.py` verifica i manifest e ricalcola le metriche
dei CSV con il validatore ufficiale e le funzioni della dashboard. Il report è
`data/cache_mv/augmentation_runs/aug_sporcate_v2/validazione_risultati.json`.
Con `--modelli <nome1> <nome2>` verifica anche la rigenerazione identica dei CSV
finali, senza leggere la ground truth pubblica. Il controllo pubblico richiede
`--pubblico` esplicito e non deve guidare scelte successive.

### Modello finale con augmentation per Barça–Athletic

Dopo il confronto completo, esegui nell'ambiente `mtah`:

```bash
python modelli_mv/finale_augmentation.py --esperimento aug_sporcate_v2 --modo combinato
```

In questo confronto `combinato` è la ricetta selezionata sulle partite esterne.
Per un nuovo artefatto, omettendo `--modo`, la selezione è automatica e usa soltanto
i risultati esterni. Il comando calibra il Viterbi sulle
predizioni fuori gruppo DFL + Metrica, poi allena HGB e le reti sulle 30 partite sporcate.
Non forza più la ricetta combinata. Il numero di iterazioni HGB è la mediana dei cinque
modelli HGB della ricetta scelta. A pesi e parametri fissati, scrive
`evaluation/results/aug_sporcate_v2__<ricetta>.csv`. Il punteggio pubblico viene riportato
solo a questo punto e non deve essere usato per modificare o scegliere il modello.

Per fissare esplicitamente una ricetta già valutata usare `--modo combinato` (o `base`, `tracking`,
`transizioni`). `--nome <nome_nuovo>` permette di nominare l'artefatto; `--solo-allena` ferma il comando
prima di leggere la partita di gara. I CSV esterni conservano sempre il Viterbi del rispettivo fold,
mai quello finale calibrato su tutte le predizioni OOF.

I quattro artefatti della prova completata sono stati finalizzati con `--modo` esplicito.
La ripresa richiede la stessa scelta; per rigenerare soltanto un CSV dai pesi già pronti
usare `riproduci.py prevedi`, senza rilanciare l'addestramento finale.

Risultati verificati del confronto completo (media delle dieci sporcate):

| ricetta | spellF1 | macroF1 |
|---|---:|---:|
| base | 72,05 | 93,46 |
| tracking | 71,96 | 93,34 |
| transizioni | 71,93 | 93,42 |
| combinato | 72,39 | 93,48 |

120 viste ricomputate esattamente, 40 CSV esterni validati, quattro CSV finali riproducibili
SHA-256 e 46 test superati. Il guadagno del combinato è piccolo (+0,34 punti spellF1),
non una dimostrazione di miglioramento robusto. Dettagli e controllo pubblico finale in
[`changelog.md`](../changelog.md).

I pesi sono in `data/cache_mv/modelli/aug_sporcate_v2__<ricetta>/`, con ricetta, provenienza,
manifest degli artefatti e benchmark esterno. Per rigenerare il CSV, sostituire `<ricetta>`:

```bash
python modelli_mv/riproduci.py prevedi --modello aug_sporcate_v2__<ricetta>
```

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
