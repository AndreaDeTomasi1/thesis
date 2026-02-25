#set page(
  paper: "a4",
  margin: 2.5cm,
)
#set text(lang: "it", size: 11pt, font: "Linux Libertine")

= Progettazione di un agente RAG per il supporto alla centrale di AREU (112)

== Abstract

Il presente lavoro di tesi, svolto nell’ambito del Master in Artificial Intelligence and Data Analytics for Business, ha come obiettivo lo sviluppo di un agente conversazionale basato su Retrieval-Augmented Generation (RAG) per supportare gli operatori della centrale di AREU. Il sistema consente l’accesso rapido alle istruzioni operative, migliorando l’efficienza decisionale.

== 1. Obiettivi

L’obiettivo del progetto è stato progettare un sistema RAG per:
- accesso rapido alle procedure
- riduzione del carico cognitivo
- standardizzazione delle decisioni
- supporto operativo in contesti critici

== 2. Contesto

Le centrali di emergenza richiedono rapidità decisionale e accesso immediato alle informazioni. I Large Language Models consentono nuove modalità di interazione con la documentazione.

== 3. Descrizione del progetto

=== 3.1 Architettura

Il sistema è composto da:
- indicizzazione documenti
- generazione embedding
- retrieval semantico
- generazione risposta
- orchestrazione agentica

Il sistema è stato sviluppato in Python con template Jinja e file YAML di configurazione.

=== 3.2 Stato dell'Arte: Sistemi di Retrieval-Augmented Generation (RAG)

=== Origini e contesto storico

Negli ultimi anni, i Large Language Models (LLM) hanno dimostrato elevate capacità di apprendere conoscenza fattuale e semantica da grandi quantità di dati testuali. Tuttavia, tali modelli presentano limiti strutturali significativi, tra cui la difficoltà di aggiornamento continuo, la scarsa interpretabilità e la tendenza a generare informazioni errate o non verificabili, fenomeno noto come hallucination. Questi limiti derivano dalla natura parametrica della conoscenza incorporata nei modelli, che risulta statica e difficile da modificare senza ulteriori fasi di addestramento.

Per affrontare tali problematiche, il paradigma della *Retrieval-Augmented Generation (RAG)* è stato introdotto da Lewis et al. nel 2020 @lewis2020rag. In questo approccio, la conoscenza parametrica del modello viene integrata con una memoria esterna non parametrica, costituita da collezioni documentali indicizzate e accessibili tramite tecniche di retrieval semantico. I modelli RAG combinano tipicamente un retriever neurale con un modello generativo, consentendo di recuperare informazioni rilevanti in tempo reale e migliorando le prestazioni nei task knowledge-intensive.

Questo paradigma ha dimostrato di aumentare la fattualità, la trasparenza e l’aggiornabilità dei sistemi basati su LLM, riducendo al contempo la necessità di riaddestramento continuo. Inoltre, la possibilità di tracciare le fonti informative durante la generazione rappresenta un elemento chiave per la fiducia e la verificabilità delle risposte @rag_survey2023.

=== Evoluzione dei paradigmi RAG

La letteratura recente evidenzia una progressiva evoluzione delle architetture RAG, caratterizzata da una crescente modularità e da un’integrazione sempre più stretta tra retrieval e generazione. Le prime implementazioni seguivano una pipeline lineare, in cui una query veniva trasformata in embedding, utilizzata per recuperare documenti e successivamente fornita al modello generativo @rag_survey2023.

Successivamente, la ricerca ha introdotto numerosi miglioramenti, tra cui tecniche di query expansion, selezione dinamica del contesto, re-ranking e filtraggio semantico. Tali approcci mirano a migliorare la qualità del contesto fornito al modello generativo, evidenziando come la rilevanza nei sistemi generativi differisca da quella nei sistemi di Information Retrieval tradizionali @survey2025.

Un’evoluzione importante riguarda l’integrazione tra retrieval e reasoning. Le architetture moderne includono approcci multi-hop, in cui il sistema esegue più iterazioni di recupero e ragionamento, nonché strategie di decomposizione della query. Questo consente di affrontare task complessi, che richiedono inferenze su più documenti o fonti eterogenee.

Parallelamente, si osserva un crescente interesse per l’integrazione con sistemi multimodali, database strutturati e grafi di conoscenza, ampliando il ruolo del retrieval oltre il semplice accesso a documenti testuali.

=== Componenti core e innovazioni tecniche

Un sistema RAG si basa tipicamente su tre componenti fondamentali:  
- retrieval da fonti esterne  
- integrazione della conoscenza  
- generazione del testo  

I meccanismi di retrieval si sono evoluti da approcci lessicali, come BM25, a modelli neurali densi basati su rappresentazioni vettoriali condivise tra query e documenti. L’adozione di embedding semantici consente di migliorare la robustezza e la generalizzazione dei sistemi.

Un aspetto centrale nella progettazione dei sistemi RAG riguarda l’allineamento tra retriever e generatore. In questo contesto, il lavoro OpenRAG propone un’ottimizzazione end-to-end del retriever basata sulla rilevanza in-context, dimostrando che la rilevanza per il contesto generativo differisce da quella dei sistemi di retrieval tradizionali @openrag2025. Questo approccio consente di ottenere miglioramenti significativi anche con modelli generativi di dimensioni ridotte, suggerendo che l’efficienza del retrieval può essere più determinante della scala del modello.

Inoltre, recenti studi evidenziano l’importanza della gestione della lunghezza del contesto, della compressione semantica e della selezione adattiva dei documenti, al fine di ottimizzare l’utilizzo della finestra di contesto degli LLM.

=== Frontiere recenti e nuove direzioni

La ricerca contemporanea esplora nuove direzioni per migliorare la scalabilità, la robustezza e le capacità di ragionamento dei sistemi RAG.

==== Knowledge-oriented RAG

Un filone emergente riguarda i sistemi orientati alla conoscenza, che integrano fonti eterogenee come grafi di conoscenza, database relazionali e basi di dati strutturate. Questi approcci mirano a migliorare la coerenza logica, la verificabilità e la capacità di reasoning complesso, favorendo una maggiore integrazione tra conoscenza simbolica e rappresentazioni neurali @survey2025.

==== Robustezza e rumore nel retrieval

Un risultato controintuitivo emerso in letteratura riguarda il ruolo del rumore nel processo di retrieval. Il lavoro *The Power of Noise* mostra che l’introduzione controllata di documenti non perfettamente rilevanti può migliorare l’accuratezza del modello generativo, suggerendo che un contesto più diversificato favorisca il processo inferenziale @noise2024. Questo risultato mette in discussione le metriche tradizionali di valutazione del retrieval e apre nuove prospettive nella progettazione dei sistemi RAG.

==== Efficienza, scalabilità e deployment

Un’ulteriore direzione di ricerca riguarda l’efficienza computazionale e la scalabilità. L’adozione di architetture distribuite, indicizzazione incrementale e caching semantico consente di applicare i sistemi RAG in contesti reali, caratterizzati da vincoli di latenza e costi computazionali. Inoltre, l’integrazione con sistemi agentici e workflow automatizzati rappresenta un’area in rapida crescita.

==== Sfide aperte

Nonostante i progressi, rimangono numerose sfide aperte, tra cui:
- allineamento tra retrieval e generazione  
- selezione efficiente della conoscenza  
- interpretabilità dei sistemi  
- valutazione realistica delle prestazioni  
- gestione di informazioni obsolete o contraddittorie  

La definizione di benchmark realistici e metriche di valutazione adeguate rappresenta un tema centrale per la ricerca futura @survey2025.

=== Conclusione

Il paradigma RAG si è evoluto da un semplice sistema di recupero documentale a un ecosistema complesso, modulare e orientato alla conoscenza. L’integrazione dinamica della conoscenza esterna consente di migliorare l’affidabilità, la factualità e la trasparenza dei modelli linguistici, rendendo tali sistemi sempre più adatti ad applicazioni reali in domini specialistici.

Le direzioni future includono sistemi multimodali, agenti autonomi, retrieval adattivo, integrazione con grafi di conoscenza e strategie avanzate di reasoning. Queste innovazioni potrebbero portare allo sviluppo di sistemi intelligenti più robusti, interpretabili e aggiornabili.

=== 3.3 Metodi

La documentazione è stata preprocessata e indicizzata. Gli embedding sono stati generati tramite Amazon Bedrock.

Il database utilizzato è PostgreSQL con supporto vettoriale.

=== 3.4 max_rag_results



== 4. Risultati

Il sistema ha dimostrato:
- buon retrieval semantico
- risposte contestualizzate
- riduzione dei tempi di ricerca

== 5. Conclusioni e sviluppi futuri

Possibili sviluppi:
- integrazione con sistemi sanitari
- miglioramento osservabilità
- multimodalità

== 6. Bibliografia
#bibliography("bibliografia.bib", style: "ieee")