# Possession Model (`possession_model`)

Modello di inferenza automatica del possesso palla frame-by-frame per **More than a Hack 2027 — Phase 1**, basato su **sincronizzazione multi-camera**, **cinematica della palla e filtro di controllo effettivo**, **dinamica collettiva di squadra** e **decodifica globale di sequenza (Viterbi)**.

---

## L'Idea Centrale

Nel dataset di tracking automatico (`ATD`) abbiamo due ostacoli principali:
1. **Giocatori anonimi e tracce spezzate:** ci sono fino a 46 slot per 22 giocatori, senza numeri di maglia fissi, che possono scambiarsi identità tra un fotogramma e l'altro.
2. **Palla invisibile nel ~41% della partita:** durante lanci lunghi, rinvii, cross o mischie, la palla sparisce temporaneamente dal tracking (`NaN`).

Invece di affidarsi solo al giocatore più vicino al pallone istante per istante, il modello combina i contatti palla verificati con indicatori collettivi invarianti allo scambio di identità dei singoli giocatori (velocità media e percentuale di giocatori al passo, spostamento del baricentro, ampiezza trasversale della squadra, controllo spaziale del campo) e trova la sequenza globale di azioni più coerente sull'intero tempo di gioco.

---

## Struttura dei Moduli

### 1. Sincronizzazione Multi-Camera & Filtro Anti-Replay (`data_loader.py`)
* Oltre alla camera tattica (`tactical`), allinea automaticamente al singolo fotogramma il tracking della regia TV (`broadcast`) tramite correlazione dei baricentri di squadra e della palla (`sync_broadcast_to_tactical`).
* **Filtro Geometrico Anti-Replay:** accetta la palla vista dalla camera TV solo nei fotogrammi in cui le posizioni 2D dei giocatori inquadrati coincidono (errore mediano `< 2.2 m`) con quelle della camera tattica allo stesso istante, scartando replay e primi piani.

### 2. Tracciamento Palla, Controllo Verificato & Ponti di Passaggio (`ball_tracking.py`)
* **Pulizia e ricostruzione traiettoria (`reconstruct_ball_trajectory`):** rimuove falsi positivi ottici sopra il limite fisico di velocità (`38 m/s`), fonde le due telecamere, interpola i voli brevi in campo e proietta l'istante esatto di uscita della palla oltre le linee laterali o di fondo (`exit_impulse`).
* **Controllo effettivo vs respinte/rimpalli (`detect_ball_control_and_passes`):** secondo il regolamento ufficiale, deviazioni, parate, spazzate o duelli aerei isolati senza successivo controllo della squadra non cambiano il possesso. I tocchi singoli isolati incastrati tra due tocchi della squadra avversaria (o deviati direttamente fuori) vengono attenuati e non spezzano l'azione.
* **Continuità sui passaggi e sulle sparizioni in gioco vivo:** salda l'arco temporale tra passaggi consecutivi della stessa squadra (`pass_bridge`) e i tratti in cui la palla sparisce in aria (`3.0–10.4 s`) mentre i giocatori continuano a correre e la stessa squadra mantiene il possesso prima e dopo (`occlusion_bridge`).

### 3. Dinamica Collettiva e Spaziale delle Squadre (`team_dynamics.py`)
La funzione `compute_team_dynamics` estrae per ogni fotogramma indicatori collettivi indipendenti dall'identità dei singoli slot:
* **Intensità di movimento (`rms_speed_smooth`, `walking_ratio_smooth`, `running_ratio_smooth`):** distingue il gioco fermo (giocatori fermi o al passo `< 1.45 m/s`) dal gioco vivo.
* **Flusso collettivo (`centroid_speed_smooth`, `attacking_drift_h`):** quando la palla è in volo e invisibile, lo spostamento concorde delle due squadre segnala che l'azione è viva e indica la direzione di attacco.
* **Ampiezza laterale e baricentro territoriale (`width_diff_h`, `territory_h`):** la squadra in possesso si allarga sulle fasce (escludendo i portieri), mentre chi difende si compatta centralmente.
* **Controllo spaziale (`pitch_control_h`) e assetto da calcio d'inizio (`kickoff_formation_mask`):** stima l'influenza spaziale attorno alla palla proiettando le velocità a `0.45 s` e riconosce quando le squadre sono schierate nelle proprie metà campo prima della battuta.

### 4. Tag Tattici & Geometria dei Calci Piazzati (`tactical_priors.py`)
* `compute_tactical_tag_priors` traduce le fasi attive dello `Smart Tagging` in prior morbidi di possesso (invertendo correttamente i tag difensivi `DEFENDING IN ...`).
* Localizza con precisione la finestra `[t_stop .. t_restart]` di angoli, rinvii, punizioni, rimesse e calci d'inizio agganciando `t_stop` all'uscita palla dalle righe e verificando che alla ripresa (`t_restart`) il pallone rientri effettivamente in campo.
* **Geometria delle rimesse laterali (`infer_throwin_team`):** deduce quale squadra batte la rimessa (dove il tag riporta `Team = N/A`) individuando il giocatore posizionato sulla linea laterale (`y ~ 0` o `y ~ 68 m`).

### 5. Decodifica Globale di Sequenza (`sequence_decoder.py` & `pipeline.py`)
* `compute_frame_and_transition_costs` combina tutti gli indicatori in costi per fotogramma `C_frame(t, {H, A, D})` e penalità di transizione dipendenti dal contesto (più basse sui contatti palla verificati e sugli impulsi di stop/ripresa, alte a palla lontana o invisibile).
* `decode_possession_sequence` trova la sequenza ottima globale su ciascun tempo tramite programmazione dinamica di **Viterbi** in tempo lineare `O(T)`, imponendo durate minime fisiche (`1.80 s` per un possesso controllato, `3.40 s` per un'interruzione di gioco).

---

## Esecuzione

```powershell
$env:MTAH_DATA_DIR = "C:\Users\basos\Desktop\git\data"
.\.venv\Scripts\python.exe possession_model\run.py
.\.venv\Scripts\python.exe evaluation\validate_submission.py evaluation\results\possession_v2.csv
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```
