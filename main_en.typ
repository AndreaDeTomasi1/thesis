#set page(
  paper: "a4",
  margin: 2.5cm,
)

#set text(
  font: "Times New Roman",
  size: 11pt,
)

= Design of a RAG agent to support the AREU (112) control room

== Abstract

This thesis work, carried out as part of the Master in Artificial Intelligence and Data Analytics for Business, aims to develop a conversational agent based on Retrieval-Augmented Generation (RAG) to support the operators of the AREU control room. The system allows for quick access to operational instructions, improving decision-making efficiency.

== 1. Objectives

The goal of the project was to design a RAG system for:
- quick access to procedures
- reduction of cognitive load
- standardization of decisions
- operational support in critical contexts

== 2. Context

Emergency call centers require quick decision-making and immediate access to information. Large Language Models enable new modes of interaction with documentation.

== 3. Project Description

=== 3.1 Architecture

The system consists of:
- document indexing
- embedding generation
- semantic retrieval
- response generation
- agent orchestration

The system was developed in Python with Jinja templates and YAML configuration files.

=== 3.2 State of the Art

==== Embeddings and semantics
Embeddings allow for vector representations of language.

==== Transformer, RAG, and agents
Transformers have revolutionized NLP. The RAG paradigm integrates retrieval and generation.

==== Observability
Observability is essential for monitoring performance and security.

=== 3.3 Methods

The documentation was preprocessed and indexed. The embeddings were generated using Amazon Bedrock.

The database used is PostgreSQL with vector support.

=== 3.4 max_rag_results

This parameter controls the maximum number of retrieved documents.

== 4. Results

The system demonstrated:
- good semantic retrieval
- contextualized responses
- reduction in search times

== 5. Conclusions and future developments

Possible developments:
- integration with healthcare systems
- improvement of observability
- multimodality