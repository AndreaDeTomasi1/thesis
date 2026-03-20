#import "template.typ": ottante-report, tableofcontents, remark, smallsection, disclaimer, copyright, address, contacts

#show: ottante-report.with(
  title: "Progettazione di un agente RAG per il supporto alla centrale di AREU (112)",
  subtitle: "",
  authors: "Andrea De Tomasi",
  date: "Aprile 2026",
  logo: "logoBicoccaAcademy.png",         
  left-header-content: "",
  right-header-content: "",
  unnumbered-sections: false,
)

#tableofcontents()

#import "@preview/cetz:0.3.2": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot
#let sigmoid(x, k: 8, x0: 0.5) = {
  1.0 / (1.0 + calc.exp(-k * (x - x0)))
}
#let mh = $upright("MultiHead")$
#let concat = $upright("Concat")$
#let head = $upright("head")$
#let attention = $upright("Attention")$
#let softmax = $upright("Softmax")$

= Abstract

---

= Introduzione

---

= Stato dell'Arte: Sistemi di Retrieval-Augmented Generation (RAG)

== Genesi del Paradigma RAG

Negli ultimi anni, i Large Language Models (LLM) hanno dimostrato elevate capacità di apprendere conoscenza fattuale e semantica da grandi quantità di dati testuali. Tuttavia, tali modelli presentano limiti strutturali significativi, tra cui la difficoltà di aggiornamento continuo, la scarsa interpretabilità e la tendenza a generare informazioni errate o non verificabili, fenomeno noto come "generation of hallucination". Queste limitazioni derivano dalla natura parametrica della conoscenza incorporata nei modelli: essa è codificata nei pesi come risultato di un processo di ottimizzazione su larga scala e rimane sostanzialmente statica. 

Dal punto di vista matematico, tale conoscenza corrisponde a un punto (o a una regione) di minimo della funzione di perdita nello spazio dei parametri; modificarla in modo mirato richiede quindi una nuova fase di addestramento o una ri-ottimizzazione, poiché non è possibile intervenire localmente sui singoli contenuti senza alterare l’equilibrio globale dei parametri appresi.

Per affrontare tali problematiche, il paradigma della Retrieval-Augmented Generation (RAG) è stato introdotto da Lewis et al. nel 2020 @lewis2020rag. In questo approccio, la conoscenza parametrica del modello viene affiancata da una memoria esterna non parametrica, composta da collezioni di documenti preventivamente indicizzati. 

== Architettura e Componenti di un Sistema RAG

Un sistema RAG si basa su tre componenti fondamentali:  
- retrieval da fonti esterne  
- integrazione della conoscenza  
- generazione del testo 

L’accesso alle fonti esterne di conoscenza, avviene tramite tecniche di retrieval semantico, ovvero metodi che non si limitano alla semplice corrispondenza lessicale tra parole chiave, ma rappresentano query e documenti come vettori in uno spazio semantico continuo (embedding). In questo spazio vettoriale, la similarità tra testi viene calcolata sulla base della distanza o dell’angolo tra i rispettivi vettori (ad esempio tramite similarità coseno), consentendo di recuperare documenti rilevanti anche quando non condividono esattamente gli stessi termini della query, ma ne esprimono il contenuto concettuale. In tal modo, il modello può integrare dinamicamente informazioni aggiornate o specialistiche senza dover modificare i propri parametri interni. Lo strumento tipicamente usato per questa fase di un sistema RAG è il retriever neurale, un sistema composto da uno o più encoder neurali (tipicamente basati su architetture Transformer) addestrati a mappare sia le query sia i documenti in vettori densi appartenenti a uno spazio semantico condiviso. Nella configurazione più comune, detta bi-encoder, un encoder codifica la query e un altro codifica i documenti (talvolta con pesi condivisi). Ciascun testo viene così mappato in un vettore $f(x) in RR^d$, dove la dimensione $d$ è fissata a priori.

L’architettura Transformer, introdotta da Vaswani et al. (2017) @vaswani2017attention, si basa esclusivamente su meccanismi di attenzione, eliminando completamente ricorrenze e convoluzioni. Il componente centrale è la self-attention, che consente a ciascun elemento di una sequenza di pesare dinamicamente tutti gli altri elementi, modellando dipendenze a lungo raggio in modo parallelo. Dato un insieme di query $Q$, key $K$ e value $V$, l’attenzione scalata (scaled dot-product attention) è definita come: 

$
attention(Q, K, V) = softmax((Q K^TT) / sqrt(d_k)) V
$

dove $d_k$ è la dimensione delle key. Il fattore di normalizzazione 
$1/sqrt(d_k)$ stabilizza il gradiente per grandi dimensioni vettoriali. 

Invece di applicare una singola funzione di attenzione con key, value e query di dimensione $d_upright("model")$, risulta vantaggioso proiettare linearmente 
le query, le key e le value $h$ volte tramite diverse proiezioni lineari apprese, rispettivamente verso spazi di dimensione $d_k$, $d_k$ e $d_v$. La multi-head attention consente al modello di prestare attenzione congiuntamente a informazioni provenienti da diversi sottospazi di rappresentazione in posizioni differenti. Con una singola testa di attenzione, l’operazione di media tende a inibire questa capacità:

$
mh(Q, K, V) = concat(head_1, ..., head_h) W^O
$

$
upright("dove") quad head_i = attention(Q W_i^Q, K W_i^K, V W_i^V)
$

e $W_i^Q, W_i^K, W_i^V$ e $W^O$ sono matrici parametriche apprese.
 
Grazie a questa architettura, il Transformer consente un’elaborazione 
interamente parallela delle sequenze e una modellazione efficiente 
delle dipendenze globali, ponendo le basi per i moderni modelli 
di linguaggio di grandi dimensioni.

La loss function è il criterio matematico che guida l’addestramento del retriever. In genere è di tipo contrastivo: dato un insieme di coppie query-documento rilevanti (positivi) e non rilevanti (negativi), la perdita è definita in modo da massimizzare la similarità tra i vettori delle coppie positive e minimizzare quella delle coppie negative. Formalmente, si ottimizzano i parametri dell’encoder affinché la struttura geometrica dello spazio vettoriale rifletta una nozione di rilevanza semantica. In questo modo, la funzione appresa induce una metrica nello spazio degli embedding che approssima il concetto di pertinenza rispetto alla query.

I documenti più rilevanti individuati dal retriever vengono forniti come contesto al modello generativo, che li utilizza per produrre la risposta. Questa integrazione tra recupero semantico e generazione testuale consente di arricchire la conoscenza parametrica del modello con informazioni esterne aggiornabili, migliorando le prestazioni nei compiti ad alta intensità di conoscenza.

Questo paradigma ha dimostrato di aumentare la fattualità, la trasparenza e l’aggiornabilità dei sistemi basati su LLM, riducendo al contempo la necessità di riaddestramento continuo. Inoltre, la possibilità di tracciare le fonti informative durante la generazione rappresenta un elemento chiave per la fiducia e la verificabilità delle risposte @rag_survey2023.

== Evoluzione dei paradigmi RAG

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

== Frontiere agentiche, valutazioni e sfide future

Un’ulteriore direzione di ricerca riguarda l’efficienza computazionale e la scalabilità. L’adozione di architetture distribuite, indicizzazione incrementale e caching semantico consente di applicare i sistemi RAG in contesti reali, caratterizzati da vincoli di latenza e costi. Inoltre, l’integrazione con sistemi agentici e workflow automatizzati rappresenta un’area in rapida crescita. Per sistema agentico si intende un’architettura in cui il modello linguistico non si limita a generare una risposta a partire da un singolo prompt, ma opera come un agente capace di pianificare, prendere decisioni intermedie e interagire con strumenti esterni (ad esempio motori di ricerca, database, API o moduli di calcolo). In tale configurazione, il modello può scomporre un compito complesso in sotto-obiettivi, richiamare iterativamente il modulo di retrieval, valutare i risultati ottenuti e aggiornare il proprio stato interno o il contesto operativo. L’integrazione tra RAG e sistemi agentici consente quindi di passare da una generazione statica, limitata a un singolo ciclo di recupero e risposta, a processi iterativi e orientati all’obiettivo, più adatti a scenari applicativi complessi e dinamici.

Nonostante i progressi nella definizione e nell’implementazione di sistemi RAG efficienti ed esaustivi, permangono ancora — tra le altre — le seguenti sfide::
- allineamento tra retrieval e generazione  
- selezione efficiente della conoscenza  
- interpretabilità dei sistemi  
- valutazione realistica delle prestazioni  
- gestione di informazioni obsolete o contraddittorie  
- definizione di benchmark realistici e metriche di valutazione adeguate

== Conclusione

Il paradigma RAG si è evoluto da un semplice sistema di recupero documentale a un ecosistema complesso, modulare e orientato alla conoscenza. L’integrazione dinamica della conoscenza esterna consente di migliorare l’affidabilità, la fattualità e la trasparenza dei modelli linguistici, rendendo tali sistemi sempre più adatti ad applicazioni reali in domini specialistici.

Le direzioni future includono sistemi multimodali, agenti autonomi, retrieval adattivo, integrazione con grafi di conoscenza e strategie avanzate di reasoning. Queste innovazioni potrebbero portare allo sviluppo di sistemi intelligenti più robusti, interpretabili e aggiornabili.

= Metodi

== Architettura del Sistema

=== Selezione del database e dei modelli

Per l'implementazione del sistema è stata effettuata una selezione mirata sia del database utilizzato per la memorizzazione delle informazioni sia dei modelli impiegati nelle fasi di retrieval e generazione. 
In particolare, è stato scelto di utilizzare PostgreSQL come sistema di gestione del database, esteso tramite pgvector, un'estensione che consente la memorizzazione e la ricerca efficiente di embedding vettoriali. 
Questa soluzione permette di integrare funzionalità di vector search direttamente all'interno di un database relazionale, semplificando l'infrastruttura del sistema e facilitando la gestione dei documenti e dei relativi embedding utilizzati nel processo di retrieval.

Per quanto riguarda i modelli di intelligenza artificiale, è stata utilizzata la piattaforma Amazon Bedrock, che fornisce accesso a diversi modelli fondamentali attraverso un'interfaccia unificata. 
L'impiego di questa piattaforma consente di integrare modelli per embedding e generazione mantenendo un'architettura flessibile e scalabile, facilitando allo stesso tempo la sperimentazione e la sostituzione dei modelli senza modifiche sostanziali all'infrastruttura del sistema.
Inoltre, l'impiego di Amazon Bedrock consente di gestire l'accesso ai modelli tramite l'infrastruttura Amazon Web Services, garantendo un maggiore controllo sui dati e contribuendo a preservare la riservatezza dei documenti utilizzati dal sistema, aspetto particolarmente rilevante nel contesto di questo lavoro.

=== LangGraph

Per l'implementazione e l'orchestrazione del workflow del sistema RAG è stato utilizzato LangGraph, una libreria sviluppata per la costruzione di applicazioni basate su Large Language Models attraverso strutture computazionali a grafo. 
LangGraph estende il paradigma delle pipeline lineari tipicamente utilizzate nei framework per LLM, consentendo di definire flussi di esecuzione più complessi caratterizzati da transizioni condizionali e cicli iterativi.

A differenza delle architetture sequenziali tradizionali, nelle quali le operazioni vengono eseguite secondo una pipeline statica, LangGraph rappresenta il processo come un grafo diretto composto da nodi e archi. 
In questo contesto, ciascun nodo rappresenta un'unità computazionale — ad esempio un modulo di retrieval, reasoning o generazione — mentre gli archi definiscono il flusso di dati e le condizioni di transizione tra le diverse fasi del sistema.

Il processo di elaborazione è inoltre guidato da uno "stato" condiviso del grafo (graph state), ovvero una struttura dati che contiene le informazioni rilevanti accumulate durante l'esecuzione. 
Tale stato viene passato tra i nodi e progressivamente aggiornato, permettendo al sistema di mantenere traccia del contesto corrente della richiesta e dei risultati intermedi prodotti dalle diverse componenti del workflow.

L'utilizzo di un'architettura basata su grafo consente di implementare strategie di reasoning più articolate, come iterazioni tra retrieval e generazione, controlli di qualità sulle risposte prodotte o meccanismi di fallback quando le informazioni recuperate risultano insufficienti. 
Allo stesso tempo, tale struttura garantisce un'elevata modularità: ogni nodo del grafo incapsula una specifica funzionalità del sistema, permettendo di modificare o sostituire singoli componenti — come il retriever o il modello generativo — senza alterare il workflow complessivo. 
Queste caratteristiche rendono il sistema più flessibile, facilitando sia l'adattamento dinamico durante l'esecuzione sia la sperimentazione di diverse configurazioni architetturali.

=== Descrizione pipeline

Il workflow del sistema RAG è rappresentato nell'immagine @fig:workflow, che illustra tutte le fasi del processo.
A seguito della ricezione di una query, il sistema decide se terminare la conversazione, generare una risposta diretta senza recuperare i documenti, o procedere con il retrieval.

In caso di retrieval, l'agente genera due domande semanticamente vicine a quella originale (hyper queries), fa un embedding di tutte e tre le query e utilizza la media dei tre embedding per recuperare i documenti più rilevanti dal database. 
I documenti recuperati vengono quindi forniti come contesto al modello generativo, che produce la risposta finale.

Il passo successivo consiste nella validazione della risposta generata, che viene confrontata con la domanda originale per verificare la coerenza e la correttezza delle informazioni fornite. In caso di esito negativo, il sistema può decidere di iterare nuovamente il processo di retrieval, aggiornando lo stato del grafo con le informazioni ottenute e generando nuove query per affinare ulteriormente la ricerca, oppure di terminare la conversazione se si ritiene che ulteriori tentativi non porteranno a un miglioramento significativo della risposta (ad esempio per mancanza di documentazione sufficiente).

Qualora la risposta venga ritenuta soddisfacente, il sistema termina la conversazione, restituendo all'utente finale una risposta contestualizzata e basata su fonti esterne, migliorando così l'affidabilità e la pertinenza delle informazioni fornite rispetto a un modello generativo standalone.

Tutti ciò che viene generato durante il processo, comprese le query, i documenti recuperati, le risposte prodotte e il tempo impiegato da ciascun nodo, viene memorizzato nello stato del grafo, consentendo al sistema di mantenere una traccia completa dell'interazione e di utilizzare queste informazioni per eventuali iterazioni successive o per analisi post-hoc delle prestazioni del sistema.
Inoltre tutti gli step sono affiancati da "reasoning", ovvero da una spiegazione testuale che descrive le decisioni prese dal sistema in ciascuna fase, migliorando così la trasparenza e l'interpretabilità del processo.

#v(2.0em)
#figure(
  image("/ObservableRagAgentTest-graph.png", width: 100%),
  caption: [RAG agent workflow]
) <fig:workflow>

#v(2.0em)

== Dataset, Preprocessing e Database

I documenti utilizzati per il retrieval sono stati raccolti da fonti pubbliche e private, e includono istruzioni operative, procedure interne e documenti di rilievo per la gestione delle emergenze.
Il processo di preprocessing ha coinvolto la pulizia dei dati, la normalizzazione del testo e la segmentazione in unità informative coerenti (chunking). ogni "chunk" sufficientemente lungo viene affiancato da un "summary" e da tre "hyper queries", ovvero domande generate automaticamente che rappresentano vari aspetti del contenuto del chunk, al fine di migliorare la copertura semantica durante il processo di retrieval. 

Il database è stato implementato utilizzando PostgreSQL con l'estensione pgvector, che consente di memorizzare e indicizzare efficientemente gli embedding vettoriali associati ai documenti.
Gli embedding sono stati generati utilizzando un modello di embedding disponibile su Amazon Bedrock, e memorizzati nel database insieme ai metadati dei documenti, come titolo e distinzione tra documenti pubblici e privati.

== Retrieval con reasoning

Nella fase di retrieval, il sistema genera due hyper queries a partire dalla query originale, le trasforma in embedding e utilizza la media dei tre embedding per recuperare i documenti più rilevanti dal database. Successivamente, viene fatta una valutazione del contenuto dei documenti recuperati, con l'obiettivo di identificare la rilevanza e la pertinenza delle informazioni rispetto alla query originale. Questa valutazione viene fatta da un modello, il quale affianca ad ogni "chunk" un punteggio compreso tra 0 e 1, che rappresenta la probabilità che il documento sia rilevante per la query, insieme a una spiegazione testuale che descrive le motivazioni della valutazione.

Dopo questa fase, i documenti vengono ordinati sulla base della rilevanza e scremati in modo da mantenere solo quelli più pertinenti tramite una "sigmoide" centrata nella media dei punteggi con steepness 8, al fine di ridurre il rumore e migliorare la qualità del contesto fornito al modello generativo. Solo i documenti con un punteggio di rilevanza superiore a 0.5 vengono mantenuti, mentre quelli con punteggio inferiore a 0.5 vengono scartati, come mostrato nella @fig:sigmoid.

#let sample-relevances = (0.82, 0.91, 0.45, 0.78, 0.23, 0.67, 0.55, 0.38, 0.71, 0.60)

#let mean-relevance = sample-relevances.sum() / sample-relevances.len()

#figure(
  canvas({
    plot.plot(
      size: (10, 6),
      x-label: [relevance score],
      y-label: [$sigma$(relevance)],
      x-min: 0, x-max: 1,
      y-min: 0, y-max: 1,
      x-tick-step: 0.25,
      y-tick-step: 0.25,
      {
        // rejected zone
        plot.add-fill-between(
          domain: (0, mean-relevance),
          samples: 100,
          x => sigmoid(x, k: 8, x0: mean-relevance),
          x => 0,
          style: (fill: red.lighten(70%), stroke: none),
        )

        // accepted zone
        plot.add-fill-between(
          domain: (mean-relevance, 1),
          samples: 100,
          x => sigmoid(x, k: 8, x0: mean-relevance),
          x => 0,
          style: (fill: green.lighten(70%), stroke: none),
        )

        // sigmoid centered on mean
        plot.add(
          domain: (0, 1),
          samples: 100,
          x => sigmoid(x, k: 8, x0: mean-relevance),
          style: (stroke: black),
        )

        // plot each document as a dot on the x-axis
        plot.add(
          sample-relevances.map(r => (r, 0.02)),
          mark: "o",
          mark-size: 0.15,
          style: (stroke: none, fill: black),
        )
      }
    )

    // vertical dashed line at the mean
    let mean-x = mean-relevance * 10  // maps [0,1] -> [0, size.x]
    draw.line(
      (mean-x, 0),
      (mean-x, 6),
      stroke: (paint: blue, dash: "dashed")
    )

    // label the mean
    draw.content(
      (mean-x + 0.15, 5.5),
      text(size: 8pt, fill: blue)[mean = #calc.round(mean-relevance, digits: 2)]
    )
  }),
  caption: [Sigmoid]
) <fig:sigmoid>

Infine, i documenti selezionati vengono forniti come contesto al modello generativo, mantenendo informazioni circa la pertinenza con la query originale, per permettere all'agente di integrare in modo più efficace le informazioni durante la generazione della risposta. 

== Generazione e Validazione

Il modello generativo utilizza come contesto i documenti recuperati con il punteggio di rilevanza, la query originale e le relative hyper queries, per produrre una risposta contestualizzata e basata su fonti attendibili. La generazione della risposta è guidata da un prompt jinja, progettato per incoraggiare il modello a integrare le informazioni in modo coerente e a fornire spiegazioni sulle decisioni prese durante la generazione. La risposta contiene citazioni ai documenti utilizzati, unito a un riassunto del contenuto e a una spiegazione del processo inferenziale seguito dal modello per arrivare alla risposta finale.

La fase di validazione dirige il flusso del sistema verso l'iterazione del retrieval o verso la terminazione della conversazione. Anche in questo passaggio, l'agente fornisce una spiegazione testuale che descrive le motivazioni alla base della decisione presa, migliorando così la trasparenza del processo e permettendo all'utente di comprendere le ragioni per cui il sistema ha ritenuto la risposta soddisfacente o meno.

= Risultati

Sono stati condotti test preliminari per valutare l'efficacia del sistema RAG implementato, focalizzandosi sulla qualità del retrieval semantico, qualità delle risposte generate e tempi di ricerca. I risultati ottenuti indicano che il sistema è in grado di recuperare documenti rilevanti anche in presenza di query complesse o ambigue, grazie all'utilizzo di embedding semantici e alla generazione di hyper queries. Le risposte prodotte risultano contestualizzate e basate su fonti attendibili, con una riduzione significativa dei tempi di ricerca rispetto a un approccio basato esclusivamente su modelli generativi senza retrieval.

Per l'esecuzione dei test sono stati coinvolti operatori di AREU, che hanno fornito feedback qualitativi sulla pertinenza delle risposte e sulla facilità d'uso del sistema. I risultati preliminari suggeriscono un potenziale significativo per l'applicazione di sistemi RAG in contesti di emergenza, dove la rapidità e l'affidabilità delle informazioni sono cruciali.

Sono stati confrontati due modelli generativi diversi per valutare l'impatto della qualità del modello sulla performance complessiva del sistema RAG. I risultati indicano che, sebbene entrambi i modelli siano in grado di integrare efficacemente le informazioni recuperate, il modello con capacità generative più avanzate produce risposte più coerenti e dettagliate, evidenziando l'importanza di un modello generativo di alta qualità per massimizzare i benefici del paradigma RAG.

= Discussione

Il sistema RAG implementato ha dimostrato di essere efficace nel recuperare informazioni rilevanti e nel generare risposte contestualizzate, migliorando l'affidabilità e la pertinenza delle informazioni fornite agli operatori di AREU. Tuttavia, sono emerse alcune limitazioni, tra cui la dipendenza dalla qualità del database e dei modelli utilizzati, nonché la necessità di ulteriori ottimizzazioni per gestire scenari più complessi o dinamici. Nello specifico, i documenti utilizzati per il retrieval non sono stati prodotti con l'obiettivo di essere processati da un sistema RAG, e presentano quindi una struttura non ottimale per il processo di retrieval e generazione. In particolare, la presenza di informazioni ridondanti, obsolete o non strutturate ha reso più difficile per il sistema identificare e integrare le informazioni più rilevanti, evidenziando l'importanza di un processo di curazione e organizzazione dei dati più mirato per massimizzare l'efficacia dei sistemi RAG.

I punti di forza del sistema includono la modularità dell'architettura, che consente di sostituire o aggiornare singoli componenti senza alterare l'intero workflow, e la capacità di integrare informazioni esterne in modo dinamico, migliorando la fattualità e la trasparenza delle risposte generate. Inoltre, l'utilizzo di un approccio basato su grafo ha permesso di implementare strategie di reasoning più articolate e di mantenere una traccia completa dell'interazione, facilitando iterazioni successive e analisi post-hoc delle prestazioni del sistema.

Tra gli sviluppi futuri, si prevede di esplorare l'integrazione con sistemi sanitari esistenti, al fine di migliorare ulteriormente la pertinenza e l'utilità delle risposte fornite agli operatori di AREU. Un'altra direzione di ricerca riguarda l'esplorazione della multimodalità, integrando fonti di conoscenza non testuali come immagini o dati strutturati per arricchire ulteriormente il contesto fornito al modello generativo. Infine, si prevede di condurre un confronto sistematico tra diversi modelli di embedding e generazione, al fine di identificare le configurazioni più efficaci per il contesto specifico di AREU, e di esplorare strategie di hierarchical ranking per ottimizzare ulteriormente la selezione dei documenti durante il processo di retrieval.


#bibliography("bibliography.bib", style: "ieee")