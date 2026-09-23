# Validazione tecnica del consolidamento

Verifiche eseguite il 22–23 settembre 2026 per il consolidamento della branch
`codex/evaluation-hardening`. Le prove reali del 22 settembre usano la base Git
`9c65e62` con modifiche non committate e prompt `2026-09-22.1`: quel commit da solo
non identifica il codice eseguito. I manifest conservano correttamente
`dirty=true`. Il consolidamento successivo comprende codice, fixture, lock e
documentazione, per rendere identificabile la baseline integrata tramite Git.

## Esito dei controlli

Ambiente verificato: macOS **15.7.4**, Apple Silicon, Python **3.13.7**. È stato
anche creato un virtualenv separato e installato `requirements-dev.lock`, che
include il lock runtime: **195 pacchetti** con versioni fissate, senza modifiche
alle versioni dei lock. La verifica iniziale in questo ambiente usa una copia dei soli file
versionabili, senza i documenti e gli output locali esclusi da Git.

Comando ripetibile dalla radice del repository:

```bash
bash scripts/verify_pipeline.sh
```

| Controllo effettivamente eseguito | Risultato |
|---|---|
| Black sui 15 file Python elencati nello script | Superato |
| Ruff, regole `F` e `I`, sugli stessi file | Superato |
| Mypy strict sui 7 moduli elencati nello script | Superato |
| Intera suite `pytest -q` nella working tree | **174 test superati**, 9 warning di deprecazione SWIG |
| Script completo su snapshot versionabile e ambiente ricostruito | **173 test superati, 1 saltato**, altri controlli superati |
| `pip check` sull'ambiente installato | Nessuna dipendenza incompatibile rilevata |
| `git diff --check` | Superato |
| Sintassi Bash dello script | Superato |
| CLI `--help` | Superato |
| Caricamento di `config.yaml` e dei tre profili YAML | Superato |
| Collegamenti locali nella documentazione | Superato |

Il controllo Mypy usa `--ignore-missing-imports --follow-imports=skip`: verifica i
moduli dichiarati, non tutto il grafo di import. L'orchestratore storico non è
ancora conforme a Mypy strict. Ruff non è una verifica di tutte le regole
disponibili. La ricostruzione dei lock è verificata sulla piattaforma indicata;
non dimostra compatibilità con altre versioni Python o altri sistemi operativi.

La suite non avvia servizi OCR o LLM e non scarica modelli. Una regressione storica
del valutatore legge artefatti locali Gemma quando disponibili; viene saltata se
questi dati, esclusi da Git, non sono presenti. Nell'esecuzione riportata era
disponibile e non ci sono stati test saltati. Nello snapshot senza dati locali
questo singolo test viene saltato; i casi sintetici del valutatore vengono eseguiti.
Non è un nuovo esperimento sui gold.

## Prova reale controllata

È stato usato un PDF locale di sviluppo, limitato alle pagine **20–21**, con
finestre di una pagina, soglia di 4000 caratteri, pruning TOC disabilitato ed
estrazione sequenziale. Il servizio OCR usa
`mlx-community/PaddleOCR-VL-1.5-bf16`; Ollama usa `llama3:latest`, temperatura zero
e seed 42. I modelli erano già disponibili localmente.

| Percorso | Stato | Chunk | Requisiti esportati | Durata del run |
|---|---|---|---|---|
| `extract` da PDF, ambiente di lavoro | `completed` | 2 elaborati | 11 | 157,635 s |
| `prepare-pdf`, snapshot e ambiente ricostruito | `completed` | 2 preparati, 0 elaborati da Ollama | 0 | 75,462 s |
| `extract` da cache, snapshot e ambiente ricostruito | `completed` | 2 caricati dalla cache ed elaborati | 11 | 96,460 s |

Il servizio OCR è stato arrestato prima del replay. Il replay è terminato senza
OCR attivo. Le durate sono osservazioni di questi run e non un benchmark; non
comprendono la preparazione iniziale dell'ambiente e l'avvio del servizio OCR.

Le evidenze restano locali, escluse da Git, nella cartella
`data/test/output/consolidation-20260922-sjwh7fmh/`: configurazioni, log, artefatti,
manifest e `artifact-audit.json`. I run sono identificati da:

- `run-20260922T151743.954795Z-3254472b` (PDF completo);
- `run-20260922T152047.025754Z-8880d100` (preparazione);
- `run-20260922T152348.626127Z-a13843a2` (replay).

Fingerprint registrate nei manifest:

- prompt SHA-256: `e60babb813afebb0efc2f97e5e1f5baca1138c4fc7d8f98e547b1d0072daa028`;
- digest Ollama: `365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1`;
- PDF SHA-256: `ffc16829228a3cea85280345a32a491f23fa20df05acbd3fff1bb06c6eabc55d`.

L'audit verifica i 18 record di checksum complessivi, coerenza tra stato e
statistiche, ID univoci dei chunk, appartenenza delle citazioni, pagine e immagini
raggiungibili. Codici e descrizioni Excel coincidono con il JSON; le schede HTML
contengono gli stessi testi e incorporano le immagini PNG associate alle regioni.
È stata ispezionata anche un'immagine della pagina sorgente. Il layout HTML nel
browser non è stato verificato, perché l'apertura del file locale è stata bloccata.

L'ispezione ha rilevato limiti semantici: una citazione può indicare soltanto il
titolo del requisito pur accompagnando una descrizione corretta; un requisito può
assorbire una frase introduttiva di quello successivo. I flag di revisione non
segnalano necessariamente questi casi, perché controllano la struttura delle
citazioni. Il PDF condivide esempi con il prompt: è materiale di sviluppo, non un
test indipendente di generalizzazione.

L'audit ha inoltre individuato percorsi verso artefatti non prodotti nel manifest
(per esempio il rapporto TOC con pruning disabilitato). La correzione successiva
registra solo percorsi esistenti, mantenendo l'identità della sorgente anche nei
run falliti. È verificata dalle regressioni offline; i manifest storici delle
prove reali non sono stati riscritti e l'inferenza non è stata ripetuta per questa
sola correzione dei metadati.

## Contratti coperti dalle regressioni

| Area | Casi verificati | Test principali |
|---|---|---|
| Configurazione | YAML e sezioni malformate, opzioni non supportate, limiti numerici e intervalli pagina | [test_config_loader.py](../tests/unit/test_config_loader.py) |
| Sicurezza del run | Input assenti o vuoti, nomi in collisione, output preesistente e manifest precedente intatto; flag di sovrascrittura realmente booleano | [test_run_safety.py](../tests/unit/test_run_safety.py) |
| Più documenti | Isolamento di OCR, immagini e cache; export aggregati; preparazione/estrazione con e senza batch; percorsi manifest esistenti | [test_run_safety.py](../tests/unit/test_run_safety.py) |
| Cache | Versione, conteggio, ID duplicati, tipi e coordinate invalide, coerenza segmenti, cache vuota esplicita | [test_chunk_cache.py](../tests/unit/test_chunk_cache.py) |
| Orchestrazione | Errori di modello, JSON e schema; run parziali; deduplica per documento; ID batch e statistiche | [test_requirement_extractor.py](../tests/unit/test_requirement_extractor.py) |
| Logging e CLI | Import senza file, configurazione esplicita, severità, riconfigurazione senza duplicati, errori esposti dalla CLI | [test_logging_config.py](../tests/unit/test_logging_config.py), [test_cli_main.py](../tests/unit/test_cli_main.py) |
| Percorso esistente | Parser su fixture, chunker, provenienza, prompt, client e writer | Altri test in [tests/unit](../tests/unit) |
| Valutatore | Inventario e riferimenti validi, metriche su casi sintetici, input ambigui e output non utilizzabili | [test_quality_evaluator.py](../tests/unit/test_quality_evaluator.py) |
| Prompt | JSON dei quattro few-shot analizzabile, citazioni valide e ordinate; continuazione del testo e confine di sezione nell'esempio sintetico | [test_prompt_templates.py](../tests/unit/test_prompt_templates.py) |

I test con parser e modello simulati verificano il comportamento della pipeline
e la conservazione dei dati; non dimostrano l'accuratezza delle inferenze reali.
Non è stata misurata una percentuale di copertura complessiva del codice.

## File del consolidamento e motivazione

Questa tabella riguarda l'ultimo consolidamento, non tutte le modifiche già
presenti nella working tree dal precedente lavoro di hardening.

| File | Motivo della modifica |
|---|---|
| `config/loader.py` | Rifiuto anticipato di configurazioni incoerenti |
| `src/requirement_extraction/requirement_extractor.py` | Stati terminali affidabili, isolamento documenti, deduplica e provenienza batch, manifest coerenti con gli artefatti prodotti |
| `src/requirement_extraction/run_safety.py` | Scoperta input deterministica e protezione degli output esistenti |
| `src/requirement_extraction/chunk_cache.py` | Validazione strutturale in scrittura e lettura delle cache |
| `src/utils/logging_config.py`, `src/cli/main.py` | Logging configurato dall'entrypoint ed errori operativi leggibili |
| `tests/unit/test_config_loader.py`, `test_requirement_extractor.py`, `test_cli_main.py` | Regressioni per i comportamenti corretti |
| `tests/unit/test_run_safety.py`, `test_chunk_cache.py`, `test_logging_config.py` | Nuovi test per protezioni, cache e logging |
| `tests/unit/test_quality_evaluator.py` | Test storico opzionale in assenza degli artefatti locali, senza disabilitare i casi sintetici |
| `src/llm_integration/prompt_templates.py`, `tests/unit/test_prompt_templates.py` | Correzione degli escape JSON e dell'esempio ambiguo di continuazione; versione del prompt aggiornata e regressioni |
| `.gitignore` | Eccezioni mirate per la documentazione e le tre fixture già usate dai test, prima escluse dai pattern generali |
| `README.md` | Istruzioni operative e limiti allineati al comportamento corrente |
| `docs/implementation.md` | Descrizione dell'implementazione basata sul codice |
| `docs/validation.md` | Evidenze di verifica e perimetro non verificato |
| `docs/adr/0001-source-document-report-layout.md` | Chiarimento che gli export separati per documento sono un obiettivo differito |
| `scripts/verify_pipeline.sh` | Un comando per ripetere i controlli offline |
| `LICENSE` | Conservazione della licenza MIT già presente su `main`, senza modificarne il contenuto |

Le fixture rese versionabili sono
`tests/fixtures/parsed_documents/sample_parsed_document.json`,
`tests/fixtures/paddle_examples/example_12_res.json` e
`tests/fixtures/paddle_examples/example_12.md`. I loro contenuti non sono stati
modificati. Nessuna nuova dipendenza è stata introdotta dall'ultimo consolidamento;
i due lock e i moduli di valutazione/riproducibilità erano già parte del lavoro
di hardening da includere nella baseline.

## Copertura della documentazione

La [guida di implementazione](implementation.md) copre responsabilità dei moduli,
contratti dati, preparazione PDF, chunking, cache e replay, concorrenza, provenienza,
export, ciclo di vita del run, logging, riproducibilità e limiti. Il README fornisce
i comandi operativi; questa pagina ne separa gli esempi dalle verifiche eseguite.

È documentazione architetturale e operativa, non un riferimento esaustivo di ogni
funzione. Non sono dichiarate percentuali di copertura delle docstring. I comandi
diversi dai percorsi esplicitamente provati sopra rimangono istruzioni d'uso.

## Cosa resta fuori dalla validazione

- Prove reali sull'intero corpus, più documenti, concorrenza e pruning TOC attivo;
  misure sistematiche di prestazioni, memoria e timeout dei servizi.
- Compatibilità con altre versioni Python, hardware o sistemi operativi.
- Ricomposizione dei requisiti che attraversano finestre batch: non è implementata.
  Anche il limite in caratteri dei chunk rimane morbido, non un budget di token.
- Controllo semantico delle citazioni: gli ID validi dimostrano un collegamento
  strutturale alla fonte, non che la descrizione sia corretta o completa.
- Validazione scientifica: il gold corrente è parziale. Perimetro annotato,
  completezza e trattamento dei requisiti non annotati vanno concordati con il
  professore prima di usare precision, recall e F1 come risultati globali.

Si può descrivere l'implementazione e le sue garanzie tecniche con queste evidenze.
Prima degli esperimenti serviranno un inventario popolato (quello versionato
contiene solo l'intestazione), documenti indipendenti dai few-shot e un protocollo
di valutazione compatibile con i riferimenti parziali.
