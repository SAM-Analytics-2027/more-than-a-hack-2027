# Confronto Visuale Risultati (`dashboard/`)

Mini-webapp leggera a **tema chiaro**, **100% indipendente e in sola lettura**, pensata per confrontare in modo semplice e immediato:
- la **Soluzione Esatta dei primi 5 minuti** (`evaluation/gt/BAR-ATH_possession_gt_public.csv`);
- i nostri algoritmi (`possession_v2.csv`, ecc.);
- i **4 file base** forniti dagli organizzatori (`baseline_raw`, `baseline_smooth25`, `baseline_smooth50`, `baseline_smooth125`).

> **Sicurezza totale:** legge i file `.csv` in memoria RAM, **non scrive mai su disco**, non tocca l'algoritmo e non modifica alcun dato.

---

## Come avviarla

- **Doppio click (Windows):** apri `avvia_dashboard.bat` dentro questa cartella.
- **Oppure da terminale** (dalla cartella `more-than-a-hack-2027`):
  ```powershell
  .\.venv\Scripts\python.exe dashboard\app.py
  ```

Quando generi un nuovo file `.csv` in `evaluation/results/`, ti basta aggiornare la pagina del browser (`F5`).

---

## Cosa mostra (3 sezioni semplici)

1. **Strisce Temporali del Possesso:** barre colorate allineate fotogramma per fotogramma (`H` Blu = Barcellona, `A` Arancione = Bilbao, `D` Grigio = Gioco Fermo) con striscia **Verde / Rossa** sotto ogni modello (uguale / diverso dalla Soluzione Esatta), zoom con la rotellina del mouse e conteggio in tempo reale di **azioni e cambi** per ciascuna riga.
2. **Tabella Primi 5 Minuti (confronto diretto con la Soluzione Esatta):** la prima riga gialla riporta i numeri veri al 100% (`17 azioni`, `20 cambi`, `300,0s`); ogni riga sottostante mostra per ciascun algoritmo i voti ufficiali (`0–100`), quanti fotogrammi/secondi sbaglia (divisi tra *squadra invertita* e *vivo/fermo*), quante azioni e cambi individua e di quanto si discosta dalla soluzione esatta.
3. **Tabella Intera Partita (1° Tempo + 2° Tempo):** azioni totali, durata media, cambi totali e percentuali `H / A / D` su tutti i 158.857 fotogrammi valutati della gara.

---

## Seconda pagina: Partite Esterne (`/esterne`)

Il pulsante **Partite esterne (DFL + Metrica)** in alto apre il confronto sulle **10 partite esterne sporcate** (7 DFL + 3 Metrica): circa 16 ore di gioco e 4.000 possessi veri, contro i 17 dei 5 minuti pubblici. È la pagina su cui **scegliere** tra i modelli; i 5 minuti servono a controllare che il file del Barça sia giusto.

**Come si giudica:** partite intere, **etichette originali** delle versioni pulite (`data/altre_partite/<fonte>_pulite/<id>/<id>_possession_gt.csv`), stesso codice ufficiale del punteggio (`evaluation/possession_metrics.py`), con un frame `X` tra i due tempi come nell'intervallo del Barça.

**Come si aggiunge un modello:** una cartella per modello in `data/risultati_esterne/<nome_modello>/`, con un file per partita (`J03WMX.csv`, …, `Sample_Game_3.csv`) nello stesso formato del file del Barça (`frame,label`, una riga per frame della partita). Ogni partita va prevista da un modello che **non l'ha vista in allenamento** (né la sporcata né la pulita), lasciandola fuori a turno; altrimenti il punteggio esce gonfiato. Come per `evaluation/results/`, tutto resta in `data/` e GitHub non lo vede.

**Cosa mostra:**
1. **Riepilogo:** per ogni modello lo spellF1 medio sulle 10 partite, per fonte (DFL, Metrica), la **stabilità** (partita peggiore – migliore), il macroF1 e in quante partite è il migliore.
2. **Partita per partita:** lo spellF1 di ogni modello su ogni partita (bordo verde = migliore della partita); clic su una colonna per aprirla.
3. **Strisce temporali** della partita scelta, come nella prima pagina, con la striscia verde/rossa su tutta la partita.
4. **Tabella della partita scelta**, con le stesse colonne della tabella dei 5 minuti ma sull'intera partita.
