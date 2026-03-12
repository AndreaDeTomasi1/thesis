#set page(
  paper: "a4",
  margin: 2.5cm,
  footer: context [
    #set align(center)
    Page #counter(page).display()
  ]
)

#set text(lang: "en", size: 11pt, font: "Georgia")
#set figure(supplement: [Figure])
#set figure(gap: 2em)

= Designing a RAG Agent for Support to the AREU Central (112)

== Abstract

---

== Introduction

---

== 1. State of the Art: Retrieval-Augmented Generation (RAG) Systems

=== 1.1 Genesis of the RAG Paradigm

In recent years, Large Language Models (LLMs) have demonstrated high capabilities in learning factual and semantic knowledge from large amounts of textual data. However, these models exhibit significant structural limitations, including the difficulty of continuous updating, poor interpretability, and the tendency to generate incorrect or unverifiable information, a phenomenon known as "generation of hallucination." These limitations stem from the parametric nature of the knowledge embedded in the models: it is encoded in the weights as a result of a large-scale optimization process and remains substantially static.

From a mathematical perspective, such knowledge corresponds to a point (or region) of minimum of the loss function in the parameter space; thus, modifying it in a targeted manner requires a new training phase or re-optimization, as it is not possible to intervene locally on individual contents without altering the overall balance of the learned parameters.

To address these issues, the paradigm of Retrieval-Augmented Generation (RAG) was introduced by Lewis et al. in 2020 @lewis2020rag. In this approach, the parametric knowledge of the model is supplemented by an external non-parametric memory, composed of collections of pre-indexed documents.

=== 1.2 Architecture and Components of a RAG System

A RAG system is based on three fundamental components:  
- retrieval from external sources  
- integration of knowledge  
- text generation 

Access to external knowledge sources occurs through semantic retrieval techniques, which are methods that do not limit themselves to simple lexical matching between keywords, but represent queries and documents as vectors in a continuous semantic space (embedding). In this vector space, the similarity between texts is calculated based on the distance or angle between their respective vectors (for example, using cosine similarity), allowing the retrieval of relevant documents even when they do not share exactly the same terms as the query, but express its conceptual content. This way, the model can dynamically integrate updated or specialized information without having to modify its internal parameters. The tool typically used for this phase of a RAG system is the neural retriever, a system composed of one or more neural encoders (typically based on Transformer architectures) trained to map both queries and documents into dense vectors belonging to a shared semantic space. In the most common configuration, called bi-encoder, one encoder encodes the query and another encodes the documents (sometimes with shared weights). Each text is thus mapped into a vector $f(x) in RR^d$, where the dimension $d$ is fixed a priori.

The Transformer architecture, introduced by Vaswani et al. (2017) @vaswani2017attention, relies exclusively on attention mechanisms, completely eliminating recurrence and convolutions. The central component is self-attention, which allows each element of a sequence to dynamically weigh all other elements, modeling long-range dependencies in parallel. Given a set of queries $Q$, keys $K$, and values $V$, the scaled attention (scaled dot-product attention) is defined as:

$
"Attention"(Q, K, V) = "softmax"((Q K^T)/sqrt(d_k)) V
$

where $d_k$ is the dimension of the keys. The normalization factor 
$1/sqrt(d_k)$ stabilizes the gradient for large vector dimensions.

Instead of applying a single attention function with keys, values, and queries of size $d_"model"$, it is advantageous to linearly project 
the queries, keys, and values $h$ times through different learned linear projections, respectively towards spaces of size $d_k$, $d_k$, and $d_v$. Multi-head attention allows the model to attend jointly to information from different subspaces of representation in different positions. With a single attention head, the averaging operation tends to inhibit this capability:

$
"MultiHead"(Q, K, V) = "Concat"("head"_1, ..., "head"_h) W^O
$

$
"where" "head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)
$

and $W_i^Q, W_i^K, W_i^V$, and $W^O$ are learned parametric matrices.

Thanks to this architecture, the Transformer allows for entirely parallel processing of sequences and efficient modeling of global dependencies, laying the groundwork for modern large language models.

The loss function is the mathematical criterion that drives the training of the retriever. It is typically contrastive: given a set of relevant (positive) and non-relevant (negative) query-document pairs, the loss is defined to maximize the similarity between the vectors of positive pairs and minimize that of negative pairs. Formally, the parameters of the encoder are optimized so that the geometric structure of the vector space reflects a notion of semantic relevance. In this way, the learned function induces a metric in the embedding space that approximates the concept of relevance to the query.

The most relevant documents identified by the retriever are provided as context to the generative model, which uses them to produce the answer. This integration between semantic retrieval and textual generation allows enriching the parametric knowledge of the model with updatable external information, improving performance on knowledge-intensive tasks.

This paradigm has proven to enhance the factuality, transparency, and updatability of LLM-based systems, while reducing the need for continuous retraining. Furthermore, the ability to trace information sources during generation is a key element for the trust and verifiability of responses @rag_survey2023.

=== 1.3 Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, characterized by increasing modularity and a tighter integration between retrieval and generation. Early implementations followed a linear pipeline, where a query was transformed into an embedding, used to retrieve documents, and subsequently provided to the generative model @rag_survey2023.

In this initial phase, retrieval mechanisms largely derived from traditional Information Retrieval: lexical approaches such as BM25 @okapi_trec1994 were progressively complemented and then replaced by dense neural models based on shared vector representations between queries and documents. The adoption of semantic embeddings improved the robustness and generalization capability of systems.

With the maturation of the paradigm, research has introduced modular improvements aimed at optimizing the quality of the context provided to the generator. Among these are strategies for chunking documents (segmenting into coherent information units), query expansion, dynamic context selection, re-ranking, and semantic filtering. These techniques reflect the awareness that the notion of relevance in generative systems differs from that in traditional retrieval systems @survey2025.

A further evolution concerns the integration between retrieval and reasoning. 
Modern architectures include multi-hop approaches, where the system performs multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This enables addressing complex tasks that require distributed inferences across multiple documents or heterogeneous sources.

In this perspective, a central aspect becomes the alignment between the retriever and the generator. Zhou & Chen (2025) propose an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative context differs from that of traditional retrieval systems @openrag2025. This approach highlights how retrieval efficiency can significantly impact overall performance, even in the presence of smaller generative models.

At the same time, recent studies underscore the importance of managing 
the context window of LLMs. Semantic compression techniques, 
adaptive document selection, and noise control aim to optimize the use of available context. In particular, Cuconasu et al. (2024) show that appropriate filtering and weighting of retrieved documents reduce noise and improve model effectiveness @noise2024, while Cheng et al. (2025) provide a systematic overview of the latest strategies for efficiently combining and selecting documents @survey2025.

A counterintuitive result that has emerged in the literature concerns the role of noise in the retrieval process. Cuconasu et al. (2024) show that the controlled introduction of non-perfectly relevant documents can enhance the accuracy of the generative model, suggesting that a more diversified context fosters the inferential process @noise2024. A possible theoretical conjecture supporting this result is that the presence of moderately heterogeneous information acts as a form of contextual regularization: the inclusion of documents that are not strictly overlapping with the query could reduce the risk of overfitting the model to partial or redundant evidence, promoting a more robust latent representation. Furthermore, a diversified context could expand the semantic space explored by the model during generation, increasing the likelihood of activating relevant but not immediately obvious conceptual connections.

Another hypothesis is that "controlled noise" improves attention calibration: the need to discriminate between strongly and weakly relevant signals might lead the model to weigh the available information more selectively, reinforcing the integration and comparison mechanisms between sources. In this perspective, the observed positive effect would not stem from noise itself, but from its function as a stimulus for selection and inferential composition within the context window. This result challenges traditional metrics for evaluating retrieval and opens new perspectives in the design of RAG systems.

Finally, there is a growing interest in integration with multimodal systems, structured databases, and knowledge graphs, expanding the role of retrieval beyond simple access to textual documents and configuring RAG systems as hybrid infrastructures for accessing and composing knowledge.

=== 1.4 Agentic Frontiers, Assessments, and Future Challenges

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching enables the application of RAG systems in real-world contexts characterized by latency and cost constraints. Furthermore, integration with agentic systems and automated workflows represents a rapidly growing area. An agentic system refers to an architecture where the language model does not merely generate a response from a single prompt but operates as an agent capable of planning, making intermediate decisions, and interacting with external tools (e.g., search engines, databases, APIs, or computation modules). In such a configuration, the model can decompose a complex task into sub-goals, iteratively invoke the retrieval module, evaluate the obtained results, and update its internal state or operational context. The integration between RAG and agentic systems thus allows for a transition from static generation, limited to a single cycle of retrieval and response, to iterative and goal-oriented processes, better suited for complex and dynamic application scenarios.

Despite progress in defining and implementing efficient and comprehensive RAG systems, the following challenges remain — among others:  
- alignment between retrieval and generation  
- efficient knowledge selection  
- interpretability of systems  
- realistic performance evaluation  
- management of obsolete or contradictory information  
- definition of realistic benchmarks and appropriate evaluation metrics  

=== 1.5 Conclusion

The RAG paradigm has evolved from a simple document retrieval system to a complex, modular, and knowledge-oriented ecosystem. The dynamic integration of external knowledge enhances the reliability, factuality, and transparency of language models, making such systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations could lead to the development of more robust, interpretable, and updatable intelligent systems.

== 2. Methods

=== 2.1 System Architecture

*Database and Model Selection*

For the implementation of the system, a targeted selection was made of both the database used for storing information and the models employed in the retrieval and generation phases. 
In particular, PostgreSQL was chosen as the database management system, extended with pgvector, an extension that allows for the efficient storage and searching of vector embeddings. 
This solution enables the integration of vector search functionality directly within a relational database, simplifying the system's infrastructure and facilitating the management of documents and their corresponding embeddings used in the retrieval process.

Regarding artificial intelligence models, the Amazon Bedrock platform was used, which provides access to various foundational models through a unified interface. 
Using this platform allows for the integration of models for embedding and generation while maintaining a flexible and scalable architecture, simultaneously facilitating experimentation and replacement of models without substantial changes to the system infrastructure.
Moreover, the use of Amazon Bedrock allows for managing access to models via the Amazon Web Services infrastructure, ensuring greater control over data and helping to preserve the confidentiality of the documents used by the system, a particularly relevant aspect in the context of this work.

*LangGraph*

For the implementation and orchestration of the RAG system workflow, LangGraph was used, a library developed for building applications based on Large Language Models through graph-based computational structures. 
LangGraph extends the paradigm of linear pipelines typically used in LLM frameworks, allowing for the definition of more complex execution flows characterized by conditional transitions and iterative loops.

Unlike traditional sequential architectures, where operations are performed according to a static pipeline, LangGraph represents the process as a directed graph composed of nodes and edges. 
In this context, each node represents a computational unit — for example, a retrieval, reasoning, or generation module — while the edges define the flow of data and transition conditions between the different phases of the system.

The processing process is also guided by a shared "graph state", which is a data structure that contains the relevant information accumulated during execution. 
This state is passed between nodes and progressively updated, allowing the system to keep track of the current context of the request and the intermediate results produced by the various components of the workflow.

Using a graph-based architecture allows for implementing more sophisticated reasoning strategies, such as iterations between retrieval and generation, quality checks on produced responses, or fallback mechanisms when the retrieved information proves insufficient. 
At the same time, this structure ensures high modularity: each node of the graph encapsulates a specific functionality of the system, allowing for modification or replacement of individual components — such as the retriever or the generative model — without altering the overall workflow. 
These features make the system more flexible, facilitating both dynamic adaptation during execution and experimentation with different architectural configurations.

*Pipeline Description*

The workflow of the RAG system is represented in the image @fig:workflow, which illustrates all phases of the process. 
Upon receiving a query, the system decides whether to end the conversation, generate a direct response without retrieving documents, or proceed with retrieval.

In the case of retrieval, the agent generates two questions semantically close to the original one (hyper queries), embeds all three queries, and uses the average of the three embeddings to retrieve the most relevant documents from the database. 
The retrieved documents are then provided as context to the generative model, which produces the final response.

The next step involves validating the generated response, which is compared with the original question to check the coherence and correctness of the provided information. If the outcome is negative, the system may decide to iterate the retrieval process again, updating the graph state with the obtained information and generating new queries to further refine the search, or to terminate the conversation if it is believed that further attempts will not lead to a significant improvement in the response (for example, due to insufficient documentation).

If the response is deemed satisfactory, the system ends the conversation, returning to the end user a contextualized response based on external sources, thus improving the reliability and relevance of the information provided compared to a standalone generative model.

Everything generated during the process, including queries, retrieved documents, produced responses, and the time taken by each node, is stored in the graph state, allowing the system to maintain a complete record of the interaction and use this information for any subsequent iterations or for post-hoc analyses of system performance. 
Moreover, all steps are accompanied by "reasoning", that is, a textual explanation that describes the decisions made by the system at each stage, thus improving the transparency and interpretability of the process.

#v(2.0em)
#figure(
  image("/grafo_rag.drawio.svg", width: 70%),
  caption: [RAG agent workflow]
) <fig:workflow>

#v(2.0em)

=== 2.2 Dataset, Preprocessing, and Database

The documents used for retrieval were collected from public and private sources, and include operational instructions, internal procedures, and relevant documents for emergency management. 
The preprocessing process involved data cleaning, text normalization, and segmentation into coherent information units (chunking). Each sufficiently long "chunk" is accompanied by a "summary" and three "hyper queries," which are automatically generated questions representing various aspects of the chunk's content, in order to improve semantic coverage during the retrieval process.

The database was implemented using PostgreSQL with the pgvector extension, which allows for the efficient storage and indexing of vector embeddings associated with the documents. 
The embeddings were generated using an embedding model available on Amazon Bedrock and stored in the database along with the metadata of the documents, such as title and distinction between public and private documents.

=== 2.3 Retrieval with Reasoning

In the retrieval phase, the system generates two hyper queries from the original query, transforms them into embeddings, and uses the average of the three embeddings to retrieve the most relevant documents from the database. Subsequently, an evaluation of the content of the retrieved documents is made, with the aim of identifying the relevance and pertinence of the information with respect to the original query. This evaluation is performed by a model, which assigns each "chunk" a score between 0 and 1, representing the probability that the document is relevant to the query, along with a textual explanation describing the reasoning behind the evaluation.

After this phase, the documents are ranked based on relevance and filtered to retain only the most pertinent ones through a "sigmoid" centered on the mean of the scores, in order to reduce noise and improve the quality of the context provided to the generative model.

Finally, the selected documents are provided as context to the generative model, maintaining information about their relevance to the original query, to allow the agent to more effectively integrate the information during the response generation.

=== 2.4 Generation and Validation

The generative model uses as context the retrieved documents with relevance scores, the original query, and the corresponding hyper queries, to produce a contextualized response based on reliable sources. The generation of the response is guided by a jinja prompt, designed to encourage the model to integrate information coherently and provide explanations for the decisions made during generation. The response contains citations to the documents used, along with a summary of the content and an explanation of the inferential process followed by the model to arrive at the final response.

The validation phase directs the system's flow towards either the iteration of retrieval or the termination of the conversation. Even at this step, the agent provides a textual explanation that describes the reasoning behind the decision made, thus improving the transparency of the process and allowing the user to understand the reasons why the system deemed the response satisfactory or not.

== 3. Results

The system demonstrated:
- good semantic retrieval
- contextualized responses
- reduction in search times

== 4. Discussion

Possible developments:
- integration with healthcare systems
- improvement of observability
- multimodality
- comparison between embedding and generation models

#bibliography("bibliography.bib", style: "ieee")