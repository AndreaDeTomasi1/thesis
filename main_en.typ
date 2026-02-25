#set page(
  paper: "a4",
  margin: 2.5cm,
)
#set text(lang: "en", size: 11pt, font: "Linux Libertine")

= Design of a RAG Agent for Supporting the AREU (112) Control Center

== Abstract

This thesis work, carried out as part of the Master's in Artificial Intelligence and Data Analytics for Business, aims to develop a conversational agent based on Retrieval-Augmented Generation (RAG) to support the operators of the AREU control center. The system allows rapid access to operational instructions, improving decision-making efficiency.

== 1. Objectives

The project's objective was to design a RAG system for:
- rapid access to procedures
- reduction of cognitive load
- standardization of decisions
- operational support in critical contexts

== 2. Context

Emergency control centers require rapid decision-making and immediate access to information. Large Language Models enable new modes of interaction with documentation.

== 3. Project Description

=== 3.1 Architecture

The system consists of:
- document indexing
- embedding generation
- semantic retrieval
- response generation
- agent orchestration

The system was developed in Python with Jinja templates and YAML configuration files.

=== 3.2 State of the Art: Retrieval-Augmented Generation (RAG) Systems

=== Origins and Historical Context

In recent years, Large Language Models (LLMs) have demonstrated high capabilities in learning factual and semantic knowledge from large amounts of text data. However, these models have significant structural limitations, including the difficulty of continuous updates, poor interpretability, and the tendency to generate incorrect or unverifiable information, a phenomenon known as hallucination. These limitations stem from the parametric nature of the knowledge embedded in the models, which is static and difficult to modify without further training phases.

To address these issues, the paradigm of *Retrieval-Augmented Generation (RAG)* was introduced by Lewis et al. in 2020 @lewis2020rag. In this approach, the model's parametric knowledge is integrated with an external non-parametric memory composed of indexed document collections that are accessible through semantic retrieval techniques. RAG models typically combine a neural retriever with a generative model, allowing for the retrieval of relevant information in real-time and improving performance on knowledge-intensive tasks.

This paradigm has proven to increase the factuality, transparency, and updatability of LLM-based systems while reducing the need for continuous retraining. Moreover, the ability to trace information sources during generation is a key element for the trustworthiness and verifiability of responses @rag_survey2023.

=== Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, characterized by increasing modularity and a closer integration between retrieval and generation. Early implementations followed a linear pipeline, where a query was transformed into an embedding, used to retrieve documents, and subsequently provided to the generative model @rag_survey2023.

Subsequently, research introduced numerous improvements, including query expansion techniques, dynamic context selection, re-ranking, and semantic filtering. These approaches aim to enhance the quality of the context provided to the generative model, emphasizing how relevance in generative systems differs from that in traditional Information Retrieval systems @survey2025.

An important evolution concerns the integration of retrieval and reasoning. Modern architectures include multi-hop approaches, where the system executes multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This allows for tackling complex tasks that require inferences across multiple documents or heterogeneous sources.

At the same time, there is a growing interest in integration with multimodal systems, structured databases, and knowledge graphs, expanding the role of retrieval beyond simple access to text documents.

=== Core Components and Technical Innovations

A RAG system typically relies on three core components:  
- retrieval from external sources  
- knowledge integration  
- text generation  

Retrieval mechanisms have evolved from lexical approaches, such as BM25, to dense neural models based on shared vector representations between queries and documents. The adoption of semantic embeddings improves the robustness and generalization of the systems.

A central aspect in the design of RAG systems concerns the alignment between the retriever and the generator. In this context, the OpenRAG work proposes an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative context differs from that of traditional retrieval systems @openrag2025. This approach allows for significant improvements even with smaller generative models, suggesting that the efficiency of retrieval can be more decisive than the model's scale.

Moreover, recent studies highlight the importance of managing context length, semantic compression, and adaptive document selection to optimize the use of the LLM's context window.

=== Recent Frontiers and New Directions

Contemporary research explores new directions to improve the scalability, robustness, and reasoning capabilities of RAG systems.

==== Knowledge-oriented RAG

An emerging trend concerns knowledge-oriented systems that integrate heterogeneous sources such as knowledge graphs, relational databases, and structured data. These approaches aim to improve logical coherence, verifiability, and complex reasoning capabilities, fostering greater integration between symbolic knowledge and neural representations @survey2025.

==== Robustness and Noise in Retrieval

An counterintuitive finding in the literature concerns the role of noise in the retrieval process. The work *The Power of Noise* shows that the controlled introduction of documents that are not perfectly relevant can improve the accuracy of the generative model, suggesting that a more diverse context fosters the inferential process @noise2024. This finding challenges traditional metrics for evaluating retrieval and opens new perspectives in the design of RAG systems.

==== Efficiency, Scalability, and Deployment

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching enables the application of RAG systems in real-world contexts characterized by latency constraints and computational costs. Furthermore, integration with agentic systems and automated workflows represents a rapidly growing area.

==== Open Challenges

Despite progress, there remain numerous open challenges, including:
- alignment between retrieval and generation  
- efficient knowledge selection  
- interpretability of systems  
- realistic performance evaluation  
- management of obsolete or contradictory information  

Defining realistic benchmarks and appropriate evaluation metrics represents a central theme for future research @survey2025.

=== Conclusion

The RAG paradigm has evolved from a simple document retrieval system to a complex, modular, and knowledge-oriented ecosystem. The dynamic integration of external knowledge enhances the reliability, factuality, and transparency of language models, making such systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations could lead to the development of more robust, interpretable, and updatable intelligent systems.

=== 3.3 Methods

The documentation was preprocessed and indexed. Embeddings were generated using Amazon Bedrock.

The database used is PostgreSQL with vector support.

=== 3.4 max_rag_results



== 4. Results

The system demonstrated:
- good semantic retrieval
- contextualized responses
- reduction in search times

== 5. Conclusions and Future Developments

Possible developments:
- integration with healthcare systems
- improvement in observability
- multimodality

== 6. Bibliography
#bibliography("bibliografia.bib", style: "ieee")