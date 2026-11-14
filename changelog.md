# Changelog — validazione della data augmentation

## Ambito e stato

Lavoro locale, senza commit o push. Dopo la precisazione dell'utente, la consegna
è limitata a rendere verificabile la data augmentation, eseguire il confronto
esterno completo e generare i CSV. Ulteriori esperimenti sulle architetture e
sull'ottimizzazione dei punteggi sono rimandati.

**Stato della prima prova: pipeline e CSV verificati tecnicamente; efficacia
dell'augmentation non dimostrata.** Il confronto `aug_sporcate_v2` è terminato, ma non giustifica la
promozione del modello rispetto al congelato. Risultati misurati e file prodotti
nelle sezioni successive; revisione metodologica del 13 novembre 2026 in fondo.

**Nuova fase `aug_pulite_v3`: confronto e verifiche completati; risultati in fondo.**
La generazione da pulite e i CSV sono utilizzabili tecnicamente; questa ricetta
non ha dimostrato un miglioramento predittivo. Media delle tre repliche sulle
dieci partite: spellF1 73,89 contro 74,00 della base. Restano discrepanze
documentate nelle occlusioni. Nessuna promozione dell'augmentation a riferimento.

## Vincoli applicati

- Lette le regole in `rules/`, con precedenza a `CLARIFICATIONS.md`.
- Gli input del modello durante il training devono essere sporcati e le etichette
  provengono dai dati puliti. In `aug_sporcate_v2` sono stati letti esclusivamente
  gli ATD delle cartelle `*_sporcate`, usando dalle `*_pulite` soltanto le etichette:
  questa è una scelta dell'esperimento, **non un divieto del regolamento**.
  Le coordinate pulite esterne possono essere usate come sorgente per generare
  nuove sporcature da presentare al modello.
- Nessun training, calibrazione o scelta sul Barça, sul suo video o sui cinque
  minuti pubblici. Nessun confronto delle predizioni con il video.
- I cinque minuti sono stati misurati dopo aver fissato tutti i modelli e parametri.
- Obiettivo esterno: media aritmetica per partita sulle sette DFL e tre Metrica
  sporcate. SkillCorner è utilizzato nel training e riportato separatamente.
- Materiali e predizioni restano in locale nelle cartelle dati già escluse da Git.
- Conservati gli artefatti congelati e i risultati preesistenti; nuove prove con
  nomi distinti. Le modifiche già presenti all'inizio non sono state annullate.

## Benchmark corretto e scelta della sorgente da rivedere

La precedente pipeline di augmentation caricava gli ATD **puliti** e valutava
anche la vista di base sul tracking pulito. Quella valutazione non misurava quindi
il comportamento sulle dieci sporcate della dashboard. Il problema era la vista
di valutazione: l'utilizzo del tracking pulito esterno come sorgente di nuove
sporcature per il training è invece consentito.

Ora `modelli_mv/augmentation.py` legge ATD/XML da `*_sporcate` e GT da
`*_pulite`. Il nome `originale` significa **sporcata di partenza, senza ulteriore
perturbazione**. Le vecchie cache non vengono riutilizzate. Questa implementazione
aggiunge rumore a dati già degradati; non dimostra che la distribuzione delle
perturbazioni sia appropriata né che sia preferibile a ripartire dalle pulite.

## Modifiche ai dati e alla cache

- `augmentation.py`: whitelist delle sole trenta partite esterne; viste e fps
  validati; hash SHA-256 degli input e del codice; versioni numeriche e manifest
  `provenienza.json`; scritture atomiche.
- Dropout di giocatori corretto: non elimina l'ultimo giocatore visibile in un
  frame successivo, evitando che le diverse viste abbiano confini temporali
  incompatibili. Nessun utilizzo delle etichette per decidere le perturbazioni.
- Trasformazioni deterministiche e non distruttive, confinate a tempi e tratti
  osservabili. Le viste non ricostruiscono osservazioni mancanti.
- `dati.py`: cache identificata anche da input e costanti, modalità legacy
  separata, niente preparazione implicita del Barça durante il training esterno,
  gestione di liste vuote e nomi non riconosciuti.
- `formato_comune.py`: controlli su XML con namespace, squadre, fps, campo,
  confini dei tempi, frame duplicati/fuori intervallo ed etichette; coordinate
  incomplete o non finite trattate come mancanti.

## Correzioni necessarie al percorso neurale e temporale

- `rete.py`: rimossa la cache delle sequenze identificata soltanto dal nome della
  partita, che poteva mescolare viste diverse della stessa gara.
- Nuova inferenza versione 2: copertura dei tratti sotto sessanta secondi e degli
  ultimi frame non campionati dallo stride. Vecchi checkpoint in modalità 1.
- Crop con contesto laterale di 127 passi per lato: loss soltanto sul centro,
  contesto e padding con peso zero; supporto dei segmenti brevi.
- Normalizzazione robusta a colonne mancanti/infinite e simmetrica rispetto allo
  scambio delle squadre; esclusione delle etichette negative/X dalla loss;
  rifiuto di pesi invalidi. Checkpoint con versione e metadati dell'architettura.
- `caratteristiche.py`: modalità `legacy=True` per i pesi precedenti. Le feature
  nuove non propagano velocità, finestre e forward-fill attraverso interruzioni
  osservabili del tracking; filtri warning locali.
- `modello.py`: corretto il caso Viterbi con durata minima pari a un frame,
  controlli su forme/probabilità/parametri, decodifica per tratti contigui e
  separazione dei tempi nel punteggio. Verificata l'ottimalità su enumerazione
  indipendente di tutte le sequenze di piccoli casi sintetici.
- `esperimento.py`: fold indipendenti dall'ordine degli input, selezione di sole
  etichette valide, probabilità allineate H/A/D anche se manca una classe,
  aggregazione per partita e scoring con separazione dei tempi.
- `riproduci.py`: routing delle feature coerente con i metadati del modello,
  scrittura atomica dei CSV, fallimento esplicito se il validatore ufficiale
  rifiuta la submission, possibilità di rigenerare senza leggere la GT pubblica.

Queste correzioni sono condivise da tutte le quattro ricette del nuovo confronto:
il confronto `base` contro le ricette aumentate usa quindi lo stesso codice.

## Protocollo di confronto e modello finale

- `prova_augmentation.py`: cinque fold a partite intere; train, calibrazione e
  test disgiunti. Le viste di una gara restano sempre nella stessa partizione.
- Quattro ricette: `base`, `tracking`, `transizioni`, `combinato`.
- Probabilità delle viste di training: 50% sporcata di base, 25% occlusioni,
  25% perturbazione di posizione; stesso budget di campionamento.
- La ricetta transizioni concentra il 25% dei crop attorno ai cambi validi.
- Ogni ricetta calibra il proprio Viterbi sulla sporcata di calibrazione DFL e
  Metrica. Prima il vincolo macroF1 0,90; soddisfatto questo, massimo spellF1.
- Stress test separato: non sceglie Viterbi o ricetta finale.
- Salvataggio di probabilità, split, metriche e impronte degli artefatti.
  CSV esterni prodotti automaticamente dalle sole predizioni test dei fold.
- `finale_augmentation.py`: non forza più `combinato`; seleziona sulla media
  esterna, oppure accetta una ricetta esplicita già valutata. Verifica confronto
  completo, dati, codice e artefatti; calibra il decoder finale sulle OOF esterne,
  allena sulle trenta sporcate e salva la provenienza.
- Il decoder finale **non** viene riutilizzato per gonfiare il punteggio delle
  esterne: i CSV della dashboard mantengono il decoder del rispettivo fold.
- La selezione della ricetta avviene sulla cross-validation di sviluppo: non è
  una stima su un ulteriore test indipendente dalla selezione.

## Verifiche e strumenti aggiunti

- `tests/test_augmentation.py`: nove regressioni su provenienza, determinismo,
  immutabilità, assenza di dipendenza dalle etichette, gap e cache.
- `tests/test_neural_features.py`: quattordici regressioni su simmetria, gap,
  parser, inferenza, crop, normalizzazione e checkpoint.
- `tests/test_decoder.py`: sette regressioni, inclusa parità con lo scorer
  ufficiale e ottimalità del decoder su casi enumerati.
- `tests/test_pipeline.py`: undici regressioni su split, calibrazione, selezione,
  manifest, OOF e parità delle metriche.
- `tests/test_validation.py`: cinque regressioni su accettazione indipendente dei
  CSV, rifiuto di frame/report errati, assenza di letture pubbliche nella verifica
  standard e ripristino del CSV originale se la rigenerazione differisce.
- `valida_augmentation.py`: controllo di tutte le trenta partite e centoventi
  viste; `--ricomputa` ricostruisce le feature dagli ATD sporcati e richiede
  uguaglianza esatta, oltre a controllare gli input e le etichette.
- `valida_risultati_augmentation.py`: CSV esterni controllati con il validatore
  ufficiale e le funzioni indipendenti della dashboard; confronto con il report
  entro 1e-10. Verifica manifest e rigenerazione identica dei CSV finali.

Risultati già verificati:

- **46/46 test superati**, log `data/cache_mv/augmentation_tests.log`.
- **30/30 partite e 120/120 viste ricomputate e validate**, con uguaglianza
  esatta delle feature, GT pulita e metadati invariati, provenienza verificata
  e nessun Inf. 4.544.569 frame per vista, 18.178.276 righe complessive.
  Report: `data/cache_mv/validazione_augmentation.json`.
- Smoke end-to-end su un fold completato: tutte le quattro ricette, una rete,
  un'epoca e HGB ridotto. Serve soltanto a verificare il percorso eseguibile.
- Verificati anche i rifiuti espliciti del comando finale: uno smoke test e il
  confronto precedente basato sul tracking pulito non possono essere promossi.
- **Riferimento congelato: dieci CSV esterni su dieci identici SHA-256** dopo
  le modifiche. SpellF1 medio 0,7406425212; macroF1 medio 0,9375781396.
  Report: `data/cache_mv/validation_frozen/report.json`.
- **Integrazione dashboard verificata:** le funzioni effettive che costruiscono
  le due pagine caricano tutti e quattro i modelli, dieci partite esterne per
  ciascuno e quattro CSV finali, senza scarti. Verifica eseguita direttamente
  sulle funzioni, senza avviare il server web. Report `dashboard_verificata.json`
  nella cartella dell'esperimento.

## Comandi per riprodurre e vedere il lavoro

Da questa cartella, nell'ambiente `mtah`:

```powershell
conda activate mtah
python -m unittest discover -s tests -v
python modelli_mv/valida_augmentation.py --ricomputa
python modelli_mv/prova_augmentation.py --nome aug_sporcate_v2 --n-reti 3 --epoche 20
python modelli_mv/finale_augmentation.py --esperimento aug_sporcate_v2 --modo combinato
python modelli_mv/valida_risultati_augmentation.py --esperimento aug_sporcate_v2 --modelli aug_sporcate_v2__base aug_sporcate_v2__tracking aug_sporcate_v2__transizioni aug_sporcate_v2__combinato
python dashboard/app.py
```

- Dashboard esterne: `http://127.0.0.1:8765/esterne`, nomi
  `aug_sporcate_v2__base`, `__tracking`, `__transizioni`, `__combinato`.
- CSV esterni: `data/risultati_esterne/aug_sporcate_v2__<ricetta>/`.
- Report e pesi dei fold: `data/cache_mv/augmentation_runs/aug_sporcate_v2/`.
- CSV finali: `evaluation/results/aug_sporcate_v2__<ricetta>.csv`.
- Per rigenerare un CSV finale: `python modelli_mv/riproduci.py prevedi --modello
  aug_sporcate_v2__<ricetta>` (sostituire il segnaposto con una ricetta allenata).
- Per confrontare esplicitamente una ricetta: aggiungere `--modo <ricetta>` al
  comando `finale_augmentation.py`. `--solo-allena` non legge il match di gara.

## Limiti da mantenere espliciti

Le etichette esterne hanno definizioni e provenienze diverse, documentate in
`data/analisi/LEGGIMI.md`: DFL traduce possesso/palla viva del fornitore, Metrica
deriva etichette dagli eventi, SkillCorner usa fasi di gioco e maschera i replay.
La sporcatura preesistente può dipendere dalle etichette esterne; le nuove
perturbazioni non vi dipendono. Nessuna etichetta è stata alterata per alzare i
punteggi. La validità del software non implica che tutte le augmentation
migliorino la precisione: l'effetto è stato misurato nel confronto completo.

La risoluzione temporale della rete resta quella esistente (stride cinque);
eventuali cambi di architettura e altri esperimenti sono rimandati come richiesto.

## Risultati conclusivi della data augmentation

**Concluso:** confronto completo su cinque fold, quattro ricette, tre reti per
ricetta/fold e venti epoche. Quaranta CSV esterni validati contro lo scorer
ufficiale e la dashboard; quattro modelli finali e quattro CSV del Barça
rigenerati identici SHA-256. Le 120 viste sono state tutte ricomputate esattamente.

| Ricetta | SpellF1 esterne | MacroF1 esterne | Delta spellF1 vs base |
|---|---:|---:|---:|
| base | 72.05 | 93.46 | +0.00 punti |
| tracking | 71.96 | 93.34 | -0.09 punti |
| transizioni | 71.93 | 93.42 | -0.12 punti |
| combinato | 72.39 | 93.48 | +0.34 punti |

Ricetta selezionata **prima** del controllo pubblico: **combinato**,
usando esclusivamente la media sulle dieci esterne sporcate.
Il confronto misura cross-validation di sviluppo, non un ulteriore test indipendente.

Controllo pubblico eseguito soltanto dopo aver fissato tutti e quattro i modelli:

| CSV finale | SpellF1 pubblico | MacroF1 pubblico |
|---|---:|---:|
| aug_sporcate_v2__base.csv | 76.36 | 94.26 |
| aug_sporcate_v2__tracking.csv | 64.21 | 94.10 |
| aug_sporcate_v2__transizioni.csv | 73.14 | 95.18 |
| aug_sporcate_v2__combinato.csv | 70.86 | 94.19 |

Questi valori pubblici non hanno determinato selezione, riaddestramenti o correzioni.
Correttezza tecnica e riproducibilità della pipeline sono verificate; il beneficio
predittivo dell'augmentation non è dimostrato dal confronto disponibile.

**Interpretazione:** il vantaggio esterno di +0,34 punti di spellF1 è piccolo e
non dimostra un miglioramento robusto senza repliche o una valutazione indipendente.
Le soglie richieste di spellF1 85 sulle esterne e 90 sul pubblico **non sono state
raggiunte**. Il combinato ha macroF1 93,48 sulle esterne e 94,19 sul pubblico;
la soglia pubblica di 95 non è quindi raggiunta dalla ricetta selezionata.
Il riferimento congelato mantiene i suoi risultati, ma il suo protocollo di
calibrazione è diverso: il confronto causale dell'augmentation è con la base
di questa stessa prova. Non sono stati cambiati modelli dopo i risultati pubblici.

### File prodotti e verifica

- `data/cache_mv/augmentation_runs/aug_sporcate_v2/risultati.json`: risultati completi per fold, fonte, partita e vista.
- `data/cache_mv/augmentation_runs/aug_sporcate_v2/validazione_risultati.json`: verifica dei 40 CSV e dei quattro modelli finali.
- `data/cache_mv/augmentation_runs/aug_sporcate_v2/dashboard_verificata.json`: caricamento delle due pagine con tutti i nuovi modelli.
- `data/cache_mv/augmentation_runs/aug_sporcate_v2/selezione_prima_del_pubblico.json`: scelta esterna fissata prima delle metriche pubbliche.
- `data/cache_mv/augmentation_runs/aug_sporcate_v2/controllo_pubblico_finale.json`: controllo finale pubblico e hash dei CSV.
- `data/cache_mv/modelli/aug_sporcate_v2__<ricetta>/`: pesi finali, parametri, ricetta, manifest, benchmark e impronte attese.
- `data/risultati_esterne/aug_sporcate_v2__<ricetta>/`: dieci CSV per ricetta, già letti dalla pagina `/esterne` della dashboard.
- `evaluation/results/aug_sporcate_v2__<ricetta>.csv`: quattro CSV completi del Barça, già letti dalla prima pagina della dashboard.

Rigenerazione dei risultati esterni: rilanciare il comando del confronto con lo stesso nome
e configurazione. I fold completati vengono verificati e i CSV riesportati senza training.
Rigenerazione di un CSV finale: `python modelli_mv/riproduci.py prevedi --modello
aug_sporcate_v2__<ricetta>` (sostituire il segnaposto).

I quattro artefatti sono stati finalizzati con `--modo` esplicito per poterli
confrontare a parità di procedura. Per riprendere `finale_augmentation.py` con
gli stessi nomi conservare lo stesso `--modo`; per la semplice rigenerazione
usare il comando `riproduci.py prevedi` sopra indicato.

Non sono stati eseguiti ulteriori esperimenti di miglioramento dopo questa validazione.
Nessun commit o push.

## Revisione metodologica — 13 novembre 2026

Questa revisione corregge l'interpretazione della consegna precedente. Non è un
nuovo esperimento: pesi, CSV, metriche e codice eseguibile non sono stati modificati.
Sono stati aggiornati questo changelog e la sezione augmentation del README.

### Criticità confermate

- Il vantaggio di +0,34 punti spellF1 del combinato è una singola realizzazione
  dell'esperimento. Tre reti nell'ensemble non equivalgono a tre repliche
  indipendenti dell'intero confronto. Non è stata stimata l'incertezza del delta;
  non è quindi dimostrato un beneficio robusto, né dimostrata l'assenza assoluta
  di beneficio. Tracking e transizioni, separatamente, hanno delta negativi.
- Le perturbazioni aggiungono occlusioni e spostamenti a input già sporcati.
  L'intensità contiene scelte euristiche (fra cui un avvio di occlusione della
  palla in media ogni otto secondi osservati), e ogni partita ha una sola vista
  per tipo. Non è stata validata la corrispondenza con una distribuzione di errori
  realistica. Aumentare soltanto il numero di semi non risolverebbe questo problema.
- Le numerosità di training nei cinque fold sono 16, 17, 19, 20 e 18 partite
  (media 18); le partite DFL/Metrica di calibrazione sono 3, 2, 1, 1 e 3.
  Il parametro `stop` scelto varia fra 2 e 25 nella ricetta base. Il protocollo
  storico allenava su 23, 23, 24, 25 e 25 partite (media 24).
- Lo scarto rispetto al congelato (spellF1 74,06) mescola numerosità di training,
  calibrazione e modifiche a feature, normalizzazione, contesto e inferenza.
  I risultati attuali non permettono di attribuire causalmente quello scarto
  all'augmentation o alle singole modifiche. Il costo di circa 0,35 punti della
  sola ricalibrazione è un controllo riferito dall'utente, non riprodotto in
  questa revisione; gli effetti non vanno assunti additivi.
- Il commento di `constants.py` e l'alias `VITERBI_PARAMS_FROZEN = START` sono
  fuorvianti. `START` contiene `stop=13`, punto iniziale della ricerca; i parametri
  effettivi del congelato, caricati da `congelato/parametri_viterbi.json`, hanno
  `stop=4`. Il percorso di inferenza congelato usa il JSON corretto. La futura
  correzione deve distinguere i due concetti: cambiare semplicemente 13 in 4
  cambierebbe anche la ricetta della ricerca. Il codice viene qui conservato
  per mantenere verificabili le impronte dell'esperimento già completato.

### Impostazione da verificare nella prossima fase, non ancora implementata

1. Usare le coordinate pulite **esterne** come sorgente e generare più sporcature
   indipendenti, mantenendo le etichette esterne e presentando al modello soltanto
   input sporcati. Verificare anche i difetti già presenti nelle sorgenti pulite.
2. Stimare e verificare frequenza e durata dei buchi, errori di posizione,
   correlazione temporale e visibilità dei giocatori sui soli dati esterni di
   training. Controllare la provenienza del generatore preesistente in
   `data/analisi/sporcatura.py`; l'eventuale dipendenza dalle etichette esterne
   può introdurre scorciatoie e va valutata esplicitamente.
3. Conservare le dieci sporcate della dashboard come benchmark fisso. Tutte le
   versioni di una stessa partita devono restare nello stesso fold; anche la
   stima dei parametri della sporcatura deve escludere la partita valutata.
4. Ripristinare una base confrontabile e misurare separatamente feature, contesto
   dei crop, normalizzazione/inferenza e augmentation, con fold, budget e semi
   appaiati. Un'ipotesi ragionevole non equivale a un miglioramento misurato.
5. Per ogni fold esterno, calibrare Viterbi mediante predizioni fuori gruppo
   generate **all'interno** delle circa 24 partite disponibili, quindi riallenare
   su tutte quelle partite prima del test esterno. Non basta filtrare le OOF
   globali: i modelli che le hanno prodotte potrebbero aver visto il test esterno.
   Anche early stopping e parametri del generatore devono rispettare lo split.
   Principio metodologico: [cross-validation annidata, documentazione
   scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
6. Ripetere il confronto con più semi di training e di sporcatura; riportare
   differenze appaiate per partita e variabilità. La selezione ripetuta sulle
   stesse dieci partite resta sviluppo: per una conferma indipendente servono
   ulteriori partite esterne mai utilizzate per scegliere. I cinque minuti
   pubblici non guidano questa revisione né le scelte successive.

## Nuova implementazione clean → dirty — 13 novembre 2026

L'utente ha autorizzato l'esecuzione della revisione. Tutto resta locale, senza
commit o push. Il codice della prova precedente è stato copiato, prima delle
modifiche, in `data/cache_mv/augmentation_runs/aug_sporcate_v2/source_snapshot_20261113/`.
È uno snapshot dei sorgenti per ricostruire quella versione, non una nuova
cartella di esecuzione autonoma. Il nuovo esperimento conserva anche i propri
sorgenti esatti in `data/cache_mv/augmentation_runs/aug_pulite_v3/sorgenti/`.

- `augmentation_realiste.py`: stima profili dalle sole coppie pulita/sporcata
  esterne del training effettivo. Usa contesti cinematici e geometrici, durata
  della visibilità/mancanza, ritardo misurato della palla e segmenti di errore
  correlato. I parametri di sporcatura non leggono le etichette. Le tracce dei
  giocatori vengono riassegnate a slot della stessa squadra dopo le interruzioni;
  la perdita dei giocatori tiene conto della visibilità della palla. Le mancanze
  presenti già nella sorgente pulita restano mancanti.
- Il riferimento del generatore è **la sporcatura sintetica esterna già presente**,
  non una misura indipendente degli errori reali di una telecamera. La provenienza
  storica di quei dati rimane quella documentata in `data/analisi/LEGGIMI.md`.
  Il nuovo codice non importa i parametri storicamente regolati sul tracking
  della partita di gara e non ricalibra usando quel tracking.
- Ogni training aumentato usa tre nuove sporcature per partita: probabilità 50%
  per l'input sporcato preesistente e 50% distribuita sulle tre repliche.
  Le pulite sono soltanto la sorgente del generatore, mai l'input del modello.
- `componenti_augmentation.py`: ripristina il percorso di training legacy e
  permette prove separate di feature, contesto dei crop, normalizzazione,
  segmentazione e inferenza. I budget di campionamento e aggiornamenti restano
  uguali. Verificata uguaglianza esatta con il training legacy su un piccolo
  caso CPU: pesi, normalizzazione e probabilità dopo tre aggiornamenti.
- HGB usa 500 iterazioni fisse per entrambe le ricette, come il budget raggiunto
  da tutti gli artefatti congelati. Nella prova precedente l'early stopping su
  partite separate aveva scelto circa 114–232 iterazioni: era un altro fattore
  confondente del confronto storico. La nuova prova non usa early stopping
  casuale per riga.
- `protocollo_augmentation.py`: per ogni fold esterno, calibrazione su tutte le
  7–9 DFL/Metrica disponibili attraverso modelli che escludono sia il fold esterno
  sia quello interno previsto. Le dieci coppie di fold esclusi sono riutilizzate
  nelle due direzioni; cinque ulteriori modelli vengono allenati sulle circa
  24 partite disponibili per prevedere i cinque fold esterni.
- `studio_augmentation.py`: confronto preregistrato `base` / `risporcata`, tre
  repliche complete (semi 2027, 12027, 22027), tre reti per ensemble e venti epoche.
  Ogni coppia base/augmentation usa gli stessi semi. Il numero di aggiornamenti
  resta identico anche nelle prove di segmentazione. I CSV sono separati per
  replica; la media dei punteggi delle repliche non viene presentata come il
  punteggio di un ensemble delle loro probabilità.
- Le sei prove sulle componenti usano soltanto la replica zero e il decoder
  annidato della base, fissato per ciascun fold. Sono confronti diagnostici
  condizionati a quel decoder, non sei nuovi modelli calibrati indipendentemente.
- Due modelli finali, uno per ricetta, usano la replica zero dichiarata prima
  dei risultati, non il seme migliore. Il decoder finale serve solo per la
  partita di gara; i CSV esterni conservano i decoder annidati. Entrambi i
  modelli devono essere fissati prima di generare i CSV del Barça.
- `constants.py`: separati `VITERBI_SEARCH_START` (`stop=13`) e
  `VITERBI_PARAMS_FROZEN` (`stop=4`). `riproduci.py` usa il primo nella ricerca e
  continua a caricare i parametri dell'artefatto dal JSON durante l'inferenza.
- Controlli su impronte di input puliti, sporcati, etichette, codice e artefatti;
  verifiche delle partizioni e della provenienza dei decoder; ripresa rifiutata
  in caso di sorgenti, cache o pesi diversi. Aggiunti test di regressione dedicati.
- `valida_studio_sporcature.py`: controlla ogni vista effettivamente usata nel
  training, la provenienza, l'integrità e l'invarianza di frame/tempi/etichette.
  `--ricomputa` ricostruisce esattamente una vista per fonte e fold esterno;
  `--diagnostica` misura separatamente la distribuzione su partite escluse,
  dopo aver fissato il generatore, senza usarla per scegliere i parametri.

### Comandi della nuova fase

Usare l'ambiente Conda anche per rendere disponibili le DLL CUDA su Windows:

```powershell
conda run --no-capture-output -n mtah python -m unittest discover -s tests -v
conda run --no-capture-output -n mtah python modelli_mv/studio_augmentation.py --nome aug_pulite_v3 --fase tutto
conda run --no-capture-output -n mtah python modelli_mv/valida_studio_sporcature.py --esperimento aug_pulite_v3 --ricomputa --finale --diagnostica
```

Il log completo è `data/cache_mv/aug_pulite_v3.log`; configurazione, piano,
probabilità, pesi, decoder, risultati e verifiche sono in
`data/cache_mv/augmentation_runs/aug_pulite_v3/`. I CSV esterni sono in
`data/risultati_esterne/aug_pulite_v3__<ricetta>_r<replica>/`. I due CSV finali
previsti sono `evaluation/results/aug_pulite_v3__base_r0.csv` e
`evaluation/results/aug_pulite_v3__risporcata_r0.csv`.

**Risultati della nuova fase: riportati nella sezione seguente.** I test
software e gli smoke non dimostrano un miglioramento predittivo. Le soglie
richieste non vengono dichiarate raggiunte senza una misura completa.

### Controlli aggiunti nella prosecuzione — 14 novembre 2026

- Ripreso l'esperimento dai 43 fit completati dopo l'interruzione della precedente
  esecuzione. Prima del riuso sono controllate le impronte dei 14 sorgenti e delle
  relative copie, degli input e degli artefatti. Il fit interrotto senza manifest
  completo viene riallenato; i fit già conclusi non vengono sostituiti.
- `valida_csv_studio.py` aggiunge un controllo separato dal training: verifica
  copertura dei 12 report e dei 120 CSV, provenienza dei decoder e uguaglianza
  esatta con la decodifica delle probabilità OOF salvate, anche nelle ablation.
  Il controllo comprende frame, etichette e SHA dei byte, senza riscrivere i CSV.
  Report: `validazione_csv_esatti.json` nella cartella dell'esperimento.
- `verify_final_lock` verifica le impronte effettive dei due modelli finali e
  dei CSV riprodotti prima e dopo il controllo pubblico. Il risultato documenta
  l'immutabilità degli artefatti, oltre all'ordine delle operazioni.
- Aggiunti 14 test dedicati al nuovo controllo, compresi file alterati, frame
  errati, probabilità non valide e modifica degli artefatti finali. La verifica
  preliminare su DFL/J03WMX ha ricostruito esattamente tutti i 145.967 frame.
  Suite completa: **118 test superati**; log `data/cache_mv/augmentation_v3_tests.log`.
- La diagnostica delle sporcature su partite escluse viene salvata separatamente
  in `diagnostica_sporcature_escluse.json`, legata con SHA alla configurazione,
  al piano e ai cinque profili. Non determina modifiche al generatore o ai modelli.
- Il completamento viene registrato solo dopo le verifiche e l'aggiornamento
  della documentazione. Il log conclusivo è `data/cache_mv/finish_aug_pulite_v3.log`.
- La prosecuzione divide il calcolo in due code con cartelle di output disgiunte:
  repliche aumentate r1/r2 e base r2 seguita dalle ablation. È un cambiamento
  dell'ordine di esecuzione, con configurazione, dati, semi, budget e sorgenti
  invariati. Un solo training aumentato viene eseguito alla volta; l'altro
  processo usa le sporcate originali. Log: `aug_pulite_v3_aug_remaining.log` e
  `aug_pulite_v3_base_ablations.log` in `data/cache_mv/`.
  La finalizzazione attende tutti i 12 report completi e verifica nuovamente
  gli artefatti tramite la fase ufficiale `finale`; il suo log è
  `data/cache_mv/finalize_after_workers_v3.log`. Il comando seriale `--fase tutto`
  resta sufficiente per riprodurre l'intero studio.
- Durante il calcolo un helper locale (`data/cache_mv/keep_awake_aug_v3.py`)
  impedisce soltanto la sospensione automatica del sistema. Lo schermo può
  spegnersi; nessuna impostazione energetica viene modificata. La richiesta
  temporanea termina al completamento dello studio o alla chiusura del processo.
- Aggiornate le istruzioni del README per verificare e trovare i CSV; rimossa
  una riga vuota superflua a fine file in `finale_augmentation.py`,
  `valida_risultati_augmentation.py` e `tests/test_pipeline.py`.

Il controllo aggiuntivo si ripete con:

```powershell
conda run --no-capture-output -n mtah python modelli_mv/valida_csv_studio.py --esperimento aug_pulite_v3
```

## Diagnostica delle sporcature su partite escluse — 14 novembre 2026

Completata una verifica descrittiva su tutte le 30 partite esterne, con tre
nuove sporcature per partita. Ciascuna partita usa il profilo della replica zero
del proprio fold esterno: quel profilo esclude la partita e tutto il suo fold.
I profili erano già fissati. Questa diagnostica non seleziona parametri, non
modifica il generatore o i modelli e non legge la ground truth della partita di gara.

Il report è
`data/cache_mv/augmentation_runs/aug_pulite_v3/diagnostica_sporcature_escluse.json`.
Contiene i 30 confronti individuali, le tre repliche, le impronte della
configurazione, del piano, dei sorgenti e dei cinque profili; la validazione
conclusiva può riutilizzarlo solo dopo aver verificato quelle impronte.
Esecuzione: 340,6 secondi, limitando a uno i thread CPU per non contendere il
training. Lo script locale che l'ha prodotto è `data/cache_mv/check_held_out_v3.py`;
il log è `data/cache_mv/check_held_out_v3.log`.

La tabella usa una media non pesata per partita all'interno della fonte: 7 DFL,
3 Metrica e 20 SkillCorner. Per la risporcatura si mediano prima le tre repliche
di ciascuna partita, poi le partite. Le durate e gli errori sono **medie dei
quantili per partita**, non quantili ottenuti raggruppando tutti i frame.
Il delta è generato meno preesistente. La frazione di palla mancante è espressa
in percentuale e il suo delta in punti percentuali; per le altre misure il delta
ha la stessa unità della misura.

| Fonte | Misura | Sporcatura preesistente | Nuova risporcatura | Delta |
|---|---|---:|---:|---:|
| DFL | Palla mancante, % | 38,581 | 34,027 | −4,554 pp |
| DFL | Buchi della palla / 90 minuti attivi | 132,091 | 158,146 | +26,055 |
| DFL | Durata buchi p25, s | 4,023 | 4,059 | +0,036 |
| DFL | Durata buchi p50, s | 8,357 | 6,942 | −1,415 |
| DFL | Durata buchi p75, s | 20,861 | 14,859 | −6,003 |
| DFL | Errore palla p50, m | 1,075 | 1,066 | −0,009 |
| DFL | Errore palla p90, m | 2,021 | 2,008 | −0,014 |
| DFL | Errore palla p99, m | 2,996 | 2,924 | −0,072 |
| DFL | Giocatori visibili medi | 21,036 | 21,086 | +0,050 |
| Metrica | Palla mancante, % | 43,032 | 43,113 | +0,080 pp |
| Metrica | Buchi della palla / 90 minuti attivi | 206,033 | 226,584 | +20,550 |
| Metrica | Durata buchi p25, s | 1,517 | 0,129 | −1,388 |
| Metrica | Durata buchi p50, s | 4,907 | 3,276 | −1,631 |
| Metrica | Durata buchi p75, s | 15,063 | 13,849 | −1,214 |
| Metrica | Errore palla p50, m | 1,120 | 1,137 | +0,017 |
| Metrica | Errore palla p90, m | 2,059 | 2,122 | +0,062 |
| Metrica | Errore palla p99, m | 3,006 | 3,120 | +0,114 |
| Metrica | Giocatori visibili medi | 21,243 | 21,270 | +0,027 |
| SkillCorner | Palla mancante, % | 22,725 | 26,022 | +3,297 pp |
| SkillCorner | Buchi della palla / 90 minuti attivi | 471,460 | 473,775 | +2,315 |
| SkillCorner | Durata buchi p25, s | 0,040 | 0,041 | +0,001 |
| SkillCorner | Durata buchi p50, s | 1,867 | 1,850 | −0,017 |
| SkillCorner | Durata buchi p75, s | 4,180 | 4,678 | +0,498 |
| SkillCorner | Errore palla p50, m | 0,649 | 0,659 | +0,010 |
| SkillCorner | Errore palla p90, m | 1,190 | 1,205 | +0,015 |
| SkillCorner | Errore palla p99, m | 1,691 | 1,708 | +0,017 |
| SkillCorner | Giocatori visibili medi | 21,482 | 21,452 | −0,030 |

Le occlusioni presentano discrepanze che impediscono di dichiarare una piena
corrispondenza con la sporcatura di riferimento:

- Tutte le sette DFL hanno meno palla mancante, con delta da −1,75 a −11,02
  punti percentuali. Il caso più distante è J03WOH: 41,59% contro 30,57%.
  I buchi diventano più frequenti e generalmente più brevi.
- Metrica riproduce bene la frazione mancante aggregata, ma non la distribuzione
  delle durate: il p25 generato è circa 0,12–0,15 secondi in ciascuno dei tre
  incontri, contro 0,99–2,24 secondi nel riferimento. La sola frazione mancante
  nasconderebbe questo scarto.
- SkillCorner ha generalmente più palla mancante. Gli scarti maggiori sono
  +7,64 punti su 2006229 e +7,28 su 2016236.

I frame attivi hanno conteggi identici fra riferimento e repliche in ogni
partita. Gli errori di posizione vengono invece misurati sui frame in cui palla
pulita e sporcata sono entrambe osservabili: le diverse sporcature non rendono
necessariamente visibili gli stessi frame. La vicinanza degli errori marginali
e del numero medio di giocatori non dimostra equivalenza delle distribuzioni
congiunte, dell'autocorrelazione o delle sequenze di identità. Queste ultime non
sono misurate direttamente in questo report.

La conclusione è **utilizzabilità tecnica con una corrispondenza soltanto
parziale delle occlusioni**. Il riferimento resta la sporcatura sintetica
esterna storica, non una misura indipendente degli errori reali di telecamera.
Non vengono assegnate soglie di accettazione a posteriori, né dichiarati
miglioramenti predittivi: l'efficacia resta affidata al confronto appaiato
completo, separato da questa diagnostica.

### Origine dei buchi brevi Metrica: controllo senza modificare il generatore

L'audit in `data/cache_mv/augmentation_runs/aug_pulite_v3/audit_durate_metrica.json`
ricostruisce gli episodi dai tre input Metrica e verifica le statistiche già
salvate. Non esegue trasformazioni, training o selezione di parametri.
Non è emerso un errore di unità o di segmentazione: input puliti/sporcati e
profili usano 25 fps; la prima soglia d'età è 5 frame, cioè 0,2 secondi.
Una segmentazione indipendente riproduce esattamente i tratti e i quantili
del riferimento. `_counts` e `_simulate` usano la stessa convenzione d'età.

È emerso invece un limite della rappresentazione statistica. Quando le
coordinate pulite tornano visibili, `_counts` ricomincia l'età da uno anche
se la palla sporcata era già mancante. Quella durata residua viene mescolata
alle durate delle occlusioni iniziate con una transizione osservata da visibile
a mancante. Delle 459 uscite dallo stato mancante prima di 5 frame, **404
(88,02%) appartengono al primo caso**, contro 55 al secondo. Tra le 404 durate
residue, 403 sono di due frame e una di tre; il ritardo misurato della palla
è di due frame in tutte e tre le partite. I conteggi indipendenti coincidono
esattamente con le statistiche delle cache.

Il pooling produce un tasso di uscita nei primi 0,2 secondi di circa
23,8–25,9% per frame, applicato dal modello anche alle nuove occlusioni. È un
meccanismo concreto che può generare buchi troppo brevi; questo controllo non
attribuisce quantitativamente a esso l'intero scarto osservato, cui possono
contribuire anche gli altri limiti della rappresentazione.

Una correzione metodologica da valutare separatamente sarebbe distinguere il
tasso di uscita degli episodi iniziali, di cui si osserva solo la durata
residua, da quello degli episodi iniziati con una transizione osservata. Anche
la simulazione dovrebbe conservare questa distinzione fino al primo cambio
di stato. **Questa correzione non è stata implementata né tarata**: il generatore
e il confronto preregistrato restano invariati. L'audit spiega un limite del
generatore attuale e non costituisce una misura di efficacia predittiva.

## Risultati verificati di aug_pulite_v3

Confronto completato: 90 fit annidati, tre repliche per ricetta; sei ablation diagnostiche su cinque fold, con riuso dei pesi per la sola inferenza.

| Ricetta | SpellF1 medio sulle repliche | MacroF1 medio sulle repliche |
|---|---:|---:|
| base | 73.995 | 93.831 |
| risporcata | 73.893 | 93.887 |

Ogni replica è valutata sulla media delle stesse dieci partite. Questa tabella media tre esperimenti, non le probabilità di un nuovo ensemble.

| Modello nella dashboard | SpellF1 esterne | MacroF1 esterne |
|---|---:|---:|
| aug_pulite_v3__ab_contesto_r0 | 74.167 | 93.865 |
| aug_pulite_v3__ab_feature_r0 | 74.010 | 93.822 |
| aug_pulite_v3__ab_inferenza_r0 | 74.238 | 93.892 |
| aug_pulite_v3__ab_moderno_r0 | 73.573 | 93.764 |
| aug_pulite_v3__ab_normalizzazione_r0 | 74.199 | 93.887 |
| aug_pulite_v3__ab_segmentazione_r0 | 74.096 | 93.892 |
| aug_pulite_v3__base_r0 | 74.238 | 93.892 |
| aug_pulite_v3__base_r1 | 73.937 | 93.789 |
| aug_pulite_v3__base_r2 | 73.810 | 93.810 |
| aug_pulite_v3__risporcata_r0 | 73.920 | 93.867 |
| aug_pulite_v3__risporcata_r1 | 73.897 | 93.872 |
| aug_pulite_v3__risporcata_r2 | 73.861 | 93.921 |

Delta spellF1 appaiato della risporcatura: -0.102 punti. Intervallo bootstrap descrittivo per partita: [-0.626, +0.430]. La variabilità dei semi e le differenze per partita sono in `risultati.json`; i dieci incontri restano un insieme di sviluppo, non una conferma indipendente.

| Controllo pubblico finale | SpellF1 | MacroF1 |
|---|---:|---:|
| aug_pulite_v3__base_r0 | 78.75 | 94.81 |
| aug_pulite_v3__risporcata_r0 | 63.78 | 92.71 |

Verificate 2790 viste effettive di training, 422,644,917 righe e 15 ricomputazioni esatte. Validati 120 CSV esterni e due CSV finali rigenerati identici. Diagnostica di distribuzione su tutte le trenta partite lasciate fuori dal profilo, con tre repliche ciascuna, in `validazione_sporcature.json`.

Verificata l'integrazione nelle due pagine della dashboard tramite le loro funzioni effettive: dodici modelli esterni completi e due CSV finali, senza scarti. Nessun server web avviato. Report `dashboard_verificata.json`.

Ricostruiti inoltre tutti i 120 CSV esterni dalle probabilita OOF salvate e dai decoder verificati: frame, etichette e byte identici. Report `validazione_csv_esatti.json`, ripetibile con `python modelli_mv/valida_csv_studio.py --esperimento aug_pulite_v3`.

Il controllo pubblico è avvenuto dopo aver fissato entrambi i modelli. Nessun modello è stato modificato sulla base di quei punteggi. Nessun commit o push.

### Interpretazione conclusiva e limiti

La validazione tecnica è conclusa; **l'efficacia dell'augmentation non è
dimostrata**. Il delta medio appaiato è −0,1023 punti spellF1 e +0,0562 punti
macroF1. Gli intervalli bootstrap descrittivi al 95% sono rispettivamente
[−0,6263; +0,4298] e [−0,0592; +0,1657] punti: entrambi comprendono zero.
I delta spellF1 delle tre repliche sono −0,3181, −0,0403 e +0,0516 punti;
la deviazione standard del delta è 0,1925 punti. Tre semi non dimostrano una
maggiore stabilità generale, anche se qui la dispersione della risporcatura
è più bassa. Nessuna ricetta viene promossa a riferimento sulla base di questi
risultati esterni; il congelato resta disponibile e invariato.

Le differenze medie spellF1 per fonte sono +0,1778 punti su DFL e −0,7559 su
Metrica. La media richiesta pesa ciascuna delle dieci partite allo stesso modo,
quindi resta −0,1023; non si seleziona a posteriori la fonte favorevole.
SkillCorner è separato dal benchmark richiesto e ha delta −0,3864 punti.

Le prove delle componenti rispondono al problema delle modifiche precedentemente
mescolate. Rispetto alla base r0, mantenendo il decoder fissato:

| Componente cambiata | Delta spellF1, punti | Delta macroF1, punti |
|---|---:|---:|
| Contesto di training | −0,0714 | −0,0274 |
| Feature | −0,2283 | −0,0705 |
| Inferenza | 0,0000 | 0,0000 |
| Normalizzazione | −0,0397 | −0,0052 |
| Segmentazione | −0,1419 | −0,0007 |
| Tutte le componenti moderne | −0,6650 | −0,1279 |

Sono diagnosi su una replica e con decoder comune, non prove di danni
riproducibili di ciascuna componente. Gli effetti non sono additivi e non
spiegano automaticamente lo scarto storico di `aug_sporcate_v2`. La sola
inferenza produce dieci CSV DFL/Metrica identici alla base, ma modifica il
risultato su SkillCorner (−0,6618 spellF1 e −0,4159 macroF1): l'effetto nullo
non si generalizza a tutte le fonti. Anche la variabilità dei decoder permane:
`stop` varia da 2 a 25 nella base e da 6 a 25 nella risporcatura. Il protocollo
corregge la separazione delle partite e la numerosità della calibrazione,
senza dimostrare parametri stabili.

**Obiettivi richiesti non raggiunti congiuntamente.** La macroF1 esterna supera
90, ma la spellF1 resta sotto 85. Il controllo pubblico finale, eseguito solo a
modelli fissati, non raggiunge 90/95: base 78,75/94,81 e risporcata 63,78/92,71.
Questi punteggi pubblici sono riportati come controllo finale e non hanno guidato
scelte, correzioni o selezione. La discrepanza nelle durate delle occlusioni,
già individuata sui dati esterni, rimane un limite della ricetta attuale.

Verifica conclusiva: 118 test superati; 46 istanze di profilo, corrispondenti
a 16 profili distinti; 2.790 viste controllate strutturalmente e tramite SHA,
con 15 viste rappresentative ricomputate interamente, non tutte le 2.790;
17.190.804 righe nei 120 CSV esterni decodificati esattamente; due CSV finali
rigenerati identici; dieci CSV congelati ancora identici. Tutte le impronte
dei sorgenti del confronto coincidono con la configurazione iniziale.
Le code di calcolo e gli helper sono terminati; anche la richiesta temporanea
che impediva la sospensione automatica è stata rilasciata.
Le cache Parquet delle viste di training effettivamente usate occupano 145,86 GiB,
misurati a fine esecuzione; dati sorgente, pesi e probabilità richiedono spazio
aggiuntivo. Non sono state installate nuove dipendenze.

### Come vedere e ripetere il risultato

Avviare `conda run --no-capture-output -n mtah python dashboard/app.py` e aprire
l'indirizzo indicato nel terminale. Nella pagina `/esterne` cercare il prefisso
`aug_pulite_v3__`: ci sono sei esperimenti principali (due ricette per tre semi)
e sei prove `ab_`. Nella prima pagina ci sono i due CSV finali r0. Le due pagine
sono state verificate attraverso le loro funzioni di costruzione dei dati;
non è stata eseguita una verifica visuale tramite un server web.

Per rigenerare i CSV finali dai pesi già fissati, nell'ambiente `mtah`:

```powershell
python modelli_mv/riproduci.py prevedi --modello aug_pulite_v3__base_r0
python modelli_mv/riproduci.py prevedi --modello aug_pulite_v3__risporcata_r0
```

Per ripetere tutti i controlli senza cambiare i modelli, usare la fase `valida`
di `studio_augmentation.py`, `valida_studio_sporcature.py` con
`--ricomputa --finale --diagnostica` e `valida_csv_studio.py`, sempre con il nome
`aug_pulite_v3` come nei comandi sopra. La fase `valida` e i due validatori non
leggono la ground truth pubblica. `completamento.json` raccoglie l'esito finale;
`validazione.json`, `validazione_sporcature.json`, `validazione_csv_esatti.json`,
`dashboard_verificata.json` e `controllo_pubblico_finale.json` conservano le
verifiche distinte nella cartella dell'esperimento.
