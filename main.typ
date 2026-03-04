#set page(
  paper: "a4",
  margin: 2.5cm,
  footer: context [
    #set align(center)
    Pag #counter(page).display()
  ]
)

#set text(lang: "it", size: 11pt, font: "Georgia")

= Progettazione di un agente RAG per il supporto alla centrale di AREU (112)

== Abstract

---

== Introduzione

---

== 1. Stato dell'Arte: Sistemi di Retrieval-Augmented Generation (RAG)

=== 1.1 Genesi del Paradigma RAG

Negli ultimi anni, i Large Language Models (LLM) hanno dimostrato elevate capacità di apprendere conoscenza fattuale e semantica da grandi quantità di dati testuali. Tuttavia, tali modelli presentano limiti strutturali significativi, tra cui la difficoltà di aggiornamento continuo, la scarsa interpretabilità e la tendenza a generare informazioni errate o non verificabili, fenomeno noto come "generation of hallucination". Queste limitazioni derivano dalla natura parametrica della conoscenza incorporata nei modelli: essa è codificata nei pesi come risultato di un processo di ottimizzazione su larga scala e rimane sostanzialmente statica. 

Dal punto di vista matematico, tale conoscenza corrisponde a un punto (o a una regione) di minimo della funzione di perdita nello spazio dei parametri; modificarla in modo mirato richiede quindi una nuova fase di addestramento o una ri-ottimizzazione, poiché non è possibile intervenire localmente sui singoli contenuti senza alterare l’equilibrio globale dei parametri appresi.

Per affrontare tali problematiche, il paradigma della Retrieval-Augmented Generation (RAG) è stato introdotto da Lewis et al. nel 2020 @lewis2020rag. In questo approccio, la conoscenza parametrica del modello viene affiancata da una memoria esterna non parametrica, composta da collezioni di documenti preventivamente indicizzati. 

=== 1.2 Architettura e Componenti di un Sistema RAG

Un sistema RAG si basa su tre componenti fondamentali:  
- retrieval da fonti esterne  
- integrazione della conoscenza  
- generazione del testo 

L’accesso alle fonti esterne di conoscenza, avviene tramite tecniche di retrieval semantico, ovvero metodi che non si limitano alla semplice corrispondenza lessicale tra parole chiave, ma rappresentano query e documenti come vettori in uno spazio semantico continuo (embedding). In questo spazio vettoriale, la similarità tra testi viene calcolata sulla base della distanza o dell’angolo tra i rispettivi vettori (ad esempio tramite similarità coseno), consentendo di recuperare documenti rilevanti anche quando non condividono esattamente gli stessi termini della query, ma ne esprimono il contenuto concettuale. In tal modo, il modello può integrare dinamicamente informazioni aggiornate o specialistiche senza dover modificare i propri parametri interni. Lo strumento tipicamente usato per questa fase di un sistema RAG è il retriever neurale, un sistema composto da uno o più encoder neurali (tipicamente basati su architetture Transformer) addestrati a mappare sia le query sia i documenti in vettori densi appartenenti a uno spazio semantico condiviso. Nella configurazione più comune, detta bi-encoder, un encoder codifica la query e un altro codifica i documenti (talvolta con pesi condivisi). Ciascun testo viene così mappato in un vettore $f(x) in RR^d$, dove la dimensione $d$ è fissata a priori.

L’architettura Transformer, introdotta da Vaswani et al. (2017) @vaswani2017attention, si basa esclusivamente su meccanismi di attenzione, eliminando completamente ricorrenze e convoluzioni. Il componente centrale è la self-attention, che consente a ciascun elemento di una sequenza di pesare dinamicamente tutti gli altri elementi, modellando dipendenze a lungo raggio in modo parallelo. Dato un insieme di query $Q$, key $K$ e value $V$, l’attenzione scalata (scaled dot-product attention) è definita come: 

$
"Attention"(Q, K, V) = "softmax"((Q K^T)/sqrt(d_k)) V
$ 

dove $d_k$ è la dimensione delle key. Il fattore di normalizzazione 
$1/sqrt(d_k)$ stabilizza il gradiente per grandi dimensioni vettoriali. 

Invece di applicare una singola funzione di attenzione con key, value e query di dimensione $d_"model"$, risulta vantaggioso proiettare linearmente 
le query, le key e le value $h$ volte tramite diverse proiezioni lineari apprese, rispettivamente verso spazi di dimensione $d_k$, $d_k$ e $d_v$. La multi-head attention consente al modello di prestare attenzione congiuntamente a informazioni provenienti da diversi sottospazi di rappresentazione in posizioni differenti. Con una singola testa di attenzione, l’operazione di media tende a inibire questa capacità:

$
"MultiHead"(Q, K, V) = "Concat"("head"_1, ..., "head"_h) W^O
$

$
"dove" "head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)
$

e $W_i^Q, W_i^K, W_i^V$ e $W^O$ sono matrici parametriche apprese.
 
Grazie a questa architettura, il Transformer consente un’elaborazione 
interamente parallela delle sequenze e una modellazione efficiente 
delle dipendenze globali, ponendo le basi per i moderni modelli 
di linguaggio di grandi dimensioni.

La loss function è il criterio matematico che guida l’addestramento del retriever. In genere è di tipo contrastivo: dato un insieme di coppie query-documento rilevanti (positivi) e non rilevanti (negativi), la perdita è definita in modo da massimizzare la similarità tra i vettori delle coppie positive e minimizzare quella delle coppie negative. Formalmente, si ottimizzano i parametri dell’encoder affinché la struttura geometrica dello spazio vettoriale rifletta una nozione di rilevanza semantica. In questo modo, la funzione appresa induce una metrica nello spazio degli embedding che approssima il concetto di pertinenza rispetto alla query.

I documenti più rilevanti individuati dal retriever vengono forniti come contesto al modello generativo, che li utilizza per produrre la risposta. Questa integrazione tra recupero semantico e generazione testuale consente di arricchire la conoscenza parametrica del modello con informazioni esterne aggiornabili, migliorando le prestazioni nei compiti ad alta intensità di conoscenza.

Questo paradigma ha dimostrato di aumentare la fattualità, la trasparenza e l’aggiornabilità dei sistemi basati su LLM, riducendo al contempo la necessità di riaddestramento continuo. Inoltre, la possibilità di tracciare le fonti informative durante la generazione rappresenta un elemento chiave per la fiducia e la verificabilità delle risposte @rag_survey2023.

=== 1.3 Evoluzione dei paradigmi RAG

La letteratura recente evidenzia una progressiva evoluzione delle architetture RAG, caratterizzata da crescente modularità e da un’integrazione sempre più stretta tra retrieval e generazione. Le prime implementazioni seguivano una pipeline lineare, in cui una query veniva trasformata in embedding, utilizzata per recuperare documenti e successivamente fornita al modello generativo @rag_survey2023.

In questa fase iniziale, i meccanismi di retrieval derivavano in larga parte dall’Information Retrieval tradizionale: approcci lessicali come BM25 @okapi_trec1994 sono stati progressivamente affiancati e poi sostituiti da modelli neurali densi, basati su rappresentazioni vettoriali condivise tra query e documenti. L’adozione di embedding semantici ha migliorato la robustezza e la capacità di generalizzazione dei sistemi.

Con la maturazione del paradigma, la ricerca ha introdotto miglioramenti modulari volti a ottimizzare la qualità del contesto fornito al generatore. Tra questi rientrano strategie di chunking dei documenti (segmentazione in unità informative coerenti), query expansion, selezione dinamica del contesto, re-ranking e filtraggio semantico. Tali tecniche riflettono la consapevolezza che la nozione di rilevanza nei sistemi 
generativi differisce da quella dei sistemi di retrieval tradizionali @survey2025.

Un’evoluzione ulteriore riguarda l’integrazione tra retrieval e reasoning. 
Le architetture moderne includono approcci multi-hop, in cui il sistema esegue più iterazioni di recupero e ragionamento, nonché strategie di decomposizione della query. Questo consente di affrontare task complessi che richiedono inferenze distribuite su più documenti o fonti eterogenee.

In tale prospettiva, un aspetto centrale diventa l’allineamento tra retriever e generatore. Zhou & Chen (2025) propongono un’ottimizzazione end-to-end del retriever basata sulla rilevanza in-context, dimostrando che la rilevanza per il contesto generativo differisce da quella dei sistemi di retrieval tradizionali @openrag2025. Questo approccio evidenzia come l’efficienza del retrieval possa incidere significativamente sulle prestazioni complessive, anche in presenza di modelli generativi di dimensioni contenute.

Parallelamente, studi recenti sottolineano l’importanza della gestione 
della finestra di contesto degli LLM. Tecniche di compressione semantica, 
selezione adattiva dei documenti e controllo del rumore mirano a ottimizzare l’uso del contesto disponibile. In particolare, Cuconasu et al. (2024) mostrano che un filtraggio e una ponderazione appropriata dei documenti recuperati riducono il rumore e migliorano l’efficacia del modello @noise2024, mentre Cheng et al. (2025) forniscono una panoramica sistematica delle strategie più recenti per combinare e selezionare i documenti in modo efficiente @survey2025.

Un risultato controintuitivo emerso in letteratura riguarda il ruolo del rumore nel processo di retrieval. Cuconasu et al. (2024) mostrano che l’introduzione controllata di documenti non perfettamente rilevanti può migliorare l’accuratezza del modello generativo, suggerendo che un contesto più diversificato favorisca il processo inferenziale @noise2024. Una possibile congettura teorica a sostegno di tale risultato è che la presenza di informazione moderatamente eterogenea agisca come forma di regularizzazione contestuale: l’inclusione di documenti non strettamente sovrapponibili alla query potrebbe ridurre il rischio di sovra-adattamento del modello a evidenze parziali o ridondanti, favorendo una rappresentazione latente più robusta. Inoltre, un contesto diversificato potrebbe ampliare lo spazio semantico esplorato dal modello durante la generazione, aumentando la probabilità di attivare connessioni concettuali pertinenti ma non immediatamente evidenti.

Un’ulteriore ipotesi è che il “rumore controllato” migliori la calibrazione dell’attenzione: la necessità di discriminare tra segnali fortemente e debolmente rilevanti potrebbe indurre il modello a pesare in modo più selettivo le informazioni disponibili, rafforzando i meccanismi di integrazione e confronto tra fonti. In questa prospettiva, l’effetto positivo osservato non deriverebbe dal rumore in sé, ma dalla sua funzione di stimolo alla selezione e composizione inferenziale all’interno della finestra di contesto. Questo risultato mette in discussione le metriche tradizionali di valutazione del retrieval e apre nuove prospettive nella progettazione dei sistemi RAG.

Infine, si osserva un crescente interesse per l’integrazione con sistemi multimodali, database strutturati e grafi di conoscenza, ampliando il ruolo del retrieval oltre il semplice accesso a documenti testuali e configurando i sistemi RAG come infrastrutture ibride per l’accesso e la composizione della conoscenza.

=== 1.4 Frontiere agentiche, valutazioni e sfide future

Un’ulteriore direzione di ricerca riguarda l’efficienza computazionale e la scalabilità. L’adozione di architetture distribuite, indicizzazione incrementale e caching semantico consente di applicare i sistemi RAG in contesti reali, caratterizzati da vincoli di latenza e costi. Inoltre, l’integrazione con sistemi agentici e workflow automatizzati rappresenta un’area in rapida crescita. Per sistema agentico si intende un’architettura in cui il modello linguistico non si limita a generare una risposta a partire da un singolo prompt, ma opera come un agente capace di pianificare, prendere decisioni intermedie e interagire con strumenti esterni (ad esempio motori di ricerca, database, API o moduli di calcolo). In tale configurazione, il modello può scomporre un compito complesso in sotto-obiettivi, richiamare iterativamente il modulo di retrieval, valutare i risultati ottenuti e aggiornare il proprio stato interno o il contesto operativo. L’integrazione tra RAG e sistemi agentici consente quindi di passare da una generazione statica, limitata a un singolo ciclo di recupero e risposta, a processi iterativi e orientati all’obiettivo, più adatti a scenari applicativi complessi e dinamici.

Nonostante i progressi nella definizione e nell’implementazione di sistemi RAG efficienti ed esaustivi, permangono ancora — tra le altre — le seguenti sfide::
- allineamento tra retrieval e generazione  
- selezione efficiente della conoscenza  
- interpretabilità dei sistemi  
- valutazione realistica delle prestazioni  
- gestione di informazioni obsolete o contraddittorie  
- definizione di benchmark realistici e metriche di valutazione adeguate

=== 1.5 Conclusione

Il paradigma RAG si è evoluto da un semplice sistema di recupero documentale a un ecosistema complesso, modulare e orientato alla conoscenza. L’integrazione dinamica della conoscenza esterna consente di migliorare l’affidabilità, la fattualità e la trasparenza dei modelli linguistici, rendendo tali sistemi sempre più adatti ad applicazioni reali in domini specialistici.

Le direzioni future includono sistemi multimodali, agenti autonomi, retrieval adattivo, integrazione con grafi di conoscenza e strategie avanzate di reasoning. Queste innovazioni potrebbero portare allo sviluppo di sistemi intelligenti più robusti, interpretabili e aggiornabili.

== 2. Metodi

La documentazione è stata preprocessata e indicizzata. Gli embedding sono stati generati tramite Amazon Bedrock.

Il database utilizzato è PostgreSQL con supporto vettoriale.

== 3. Risultati

Il sistema ha dimostrato:
- buon retrieval semantico
- risposte contestualizzate
- riduzione dei tempi di ricerca

== 4. Discussione

Possibili sviluppi:
- integrazione con sistemi sanitari
- miglioramento osservabilità
- multimodalità

== 5. Bibliografia
#bibliography("bibliography.bib", style: "ieee")