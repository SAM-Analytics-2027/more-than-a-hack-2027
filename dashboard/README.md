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
