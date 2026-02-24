#set page(
  paper: "a4",
  margin: 2.5cm,
)

#set text(
  font: "Times New Roman",
  size: 11pt,
)

= Progettazione di un agente RAG per il supporto alla centrale di AREU

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

=== 3.2 State of the Art

==== Embeddings e semantica
Gli embedding consentono rappresentazioni vettoriali del linguaggio.

==== Transformer, RAG e agenti
I Transformer hanno rivoluzionato il NLP. Il paradigma RAG integra retrieval e generazione.

==== Observability
L’osservabilità è essenziale per monitorare performance e sicurezza.

=== 3.3 Metodi

La documentazione è stata preprocessata e indicizzata. Gli embedding sono stati generati tramite Amazon Bedrock.

Il database utilizzato è PostgreSQL con supporto vettoriale.

=== 3.4 max_rag_results

Questo parametro controlla il numero massimo di documenti recuperati.

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