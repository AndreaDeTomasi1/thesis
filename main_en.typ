#set page(
  paper: "a4",
  margin: 2.5cm,
  footer: context [
    #set align(center)
    Page #counter(page).display()
  ]
)

#set text(lang: "en", size: 11pt, font: "Georgia")

= Design of a RAG Agent for Supporting the AREU (112) Call Center

== Abstract

---

== Introduction

---

== 1. State of the Art: Retrieval-Augmented Generation (RAG) Systems

=== 1.1 Genesis of the RAG Paradigm

In recent years, Large Language Models (LLMs) have demonstrated high capabilities in learning factual and semantic knowledge from large amounts of textual data. However, these models present significant structural limitations, including the difficulty of continuous updating, poor interpretability, and the tendency to generate incorrect or unverifiable information, a phenomenon known as "generation of hallucination." These limitations stem from the parametric nature of the knowledge embedded in the models: it is encoded in the weights as a result of a large-scale optimization process and remains substantially static.

From a mathematical perspective, this knowledge corresponds to a point (or a region) of minimum in the loss function within the parameter space; therefore, modifying it in a targeted manner requires a new training phase or re-optimization, as it is not possible to intervene locally on individual contents without altering the global equilibrium of the learned parameters.

To address these issues, the paradigm of Retrieval-Augmented Generation (RAG) was introduced by Lewis et al. in 2020 @lewis2020rag. In this approach, the parametric knowledge of the model is complemented by an external non-parametric memory, composed of collections of pre-indexed documents.

=== 1.2 Architecture and Components of a RAG System

A RAG system is based on three fundamental components:  
- retrieval from external sources  
- knowledge integration  
- text generation 

Access to external knowledge sources occurs through semantic retrieval techniques, which are methods that do not limit themselves to simple lexical matching between keywords but represent queries and documents as vectors in a continuous semantic space (embedding). In this vector space, the similarity between texts is calculated based on the distance or angle between the respective vectors (for example, using cosine similarity), allowing relevant documents to be retrieved even when they do not share exactly the same terms as the query but express its conceptual content. In this way, the model can dynamically integrate updated or specialized information without having to modify its internal parameters. The tool typically used for this phase of a RAG system is the neural retriever, a system composed of one or more neural encoders (typically based on Transformer architectures) trained to map both queries and documents into dense vectors belonging to a shared semantic space. In the most common configuration, called bi-encoder, one encoder encodes the query and another encodes the documents (sometimes with shared weights). Each text is thus mapped into a vector $f(x) in RR^d$, where the dimension $d$ is fixed a priori.

The Transformer architecture, introduced by Vaswani et al. (2017) @vaswani2017attention, relies exclusively on attention mechanisms, completely eliminating recurrences and convolutions. The central component is self-attention, which allows each element of a sequence to dynamically weigh all other elements, modeling long-range dependencies in a parallel manner. Given a set of queries $Q$, keys $K$, and values $V$, scaled dot-product attention is defined as:

$
"Attention"(Q, K, V) = "softmax"((Q K^T)/sqrt(d_k)) V
$

where $d_k$ is the size of the keys. The normalization factor 
$1/sqrt(d_k)$ stabilizes the gradient for large vector dimensions.

Instead of applying a single attention function with keys, values, and queries of size $d_"model"$, it is advantageous to linearly project the queries, keys, and values $h$ times through different learned linear projections, respectively into spaces of size $d_k$, $d_k$, and $d_v$. Multi-head attention allows the model to attend jointly to information coming from different representation subspaces at different positions. With a single attention head, the averaging operation tends to inhibit this capability:

$
"MultiHead"(Q, K, V) = "Concat"("head"_1, ..., "head"_h) W^O
$

$
"where" "head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)
$

and $W_i^Q, W_i^K, W_i^V$ and $W^O$ are learned parametric matrices.

Thanks to this architecture, the Transformer enables fully parallel processing of sequences and efficient modeling of global dependencies, laying the groundwork for modern large language models.

The loss function is the mathematical criterion that guides the training of the retriever. It is generally of a contrastive type: given a set of relevant (positive) and non-relevant (negative) query-document pairs, the loss is defined to maximize the similarity between the vectors of positive pairs and minimize that of negative pairs. Formally, the parameters of the encoder are optimized so that the geometric structure of the vector space reflects a notion of semantic relevance. In this way, the learned function induces a metric in the embedding space that approximates the concept of relevance concerning the query.

The most relevant documents identified by the retriever are provided as context to the generative model, which uses them to produce the response. This integration between semantic retrieval and text generation allows enriching the parametric knowledge of the model with updatable external information, improving performance in knowledge-intensive tasks.

This paradigm has been shown to increase the factuality, transparency, and updatability of systems based on LLMs, while reducing the need for continuous retraining. Moreover, the ability to trace informational sources during generation represents a key element for the trust and verifiability of responses @rag_survey2023.

=== 1.3 Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, characterized by increasing modularity and an ever-closer integration between retrieval and generation. Early implementations followed a linear pipeline, where a query was transformed into an embedding, used to retrieve documents and subsequently provided to the generative model @rag_survey2023.

In this initial phase, retrieval mechanisms largely derived from traditional Information Retrieval: lexical approaches like BM25 @okapi_trec1994 were progressively supplemented and then replaced by dense neural models based on shared vector representations between queries and documents. The adoption of semantic embeddings improved the robustness and generalization capability of the systems.

With the maturation of the paradigm, research introduced modular improvements aimed at optimizing the quality of the context provided to the generator. These include document chunking strategies (segmenting into coherent informational units), query expansion, dynamic context selection, re-ranking, and semantic filtering. Such techniques reflect the awareness that the notion of relevance in generative systems differs from that in traditional retrieval systems @survey2025.

A further evolution concerns the integration between retrieval and reasoning. Modern architectures include multi-hop approaches, where the system performs multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This allows addressing complex tasks that require distributed inferences across multiple documents or heterogeneous sources.

In this perspective, a central aspect becomes the alignment between retriever and generator. Zhou & Chen (2025) propose an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative context differs from that of traditional retrieval systems @openrag2025. This approach highlights how the efficiency of retrieval can significantly impact overall performance, even in the presence of smaller generative models.

At the same time, recent studies emphasize the importance of managing the context window of LLMs. Semantic compression techniques, adaptive document selection, and noise control aim to optimize the use of available context. In particular, Cuconasu et al. (2024) show that appropriate filtering and weighting of retrieved documents reduce noise and improve the effectiveness of the model @noise2024, while Cheng et al. (2025) provide a systematic overview of the latest strategies for efficiently combining and selecting documents @survey2025.

An intuitive result that has emerged in the literature concerns the role of noise in the retrieval process. Cuconasu et al. (2024) show that the controlled introduction of not perfectly relevant documents can improve the accuracy of the generative model, suggesting that a more diversified context fosters the inferential process @noise2024. One possible theoretical conjecture supporting this result is that the presence of moderately heterogeneous information acts as a form of contextual regularization: the inclusion of documents not strictly overlapping with the query could reduce the risk of the model overfitting to partial or redundant evidence, favoring a more robust latent representation. Moreover, a diversified context could broaden the semantic space explored by the model during generation, increasing the likelihood of activating relevant but not immediately evident conceptual connections.

Another hypothesis is that "controlled noise" improves attention calibration: the need to discriminate between strongly and weakly relevant signals could prompt the model to weigh available information more selectively, reinforcing the integration and comparison mechanisms between sources. In this view, the positive effect observed would not stem from the noise itself but from its stimulating function for selection and inferential composition within the context window. This finding challenges traditional retrieval evaluation metrics and opens new perspectives in the design of RAG systems.

Finally, there is a growing interest in integration with multimodal systems, structured databases, and knowledge graphs, expanding the role of retrieval beyond simple access to textual documents and configuring RAG systems as hybrid infrastructures for accessing and composing knowledge.

=== 1.4 Agentic Frontiers, Evaluations, and Future Challenges

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching allows RAG systems to be applied in real-world contexts characterized by latency and cost constraints. Moreover, integration with agentic systems and automated workflows represents a rapidly growing area. An agentic system refers to an architecture in which the language model does not merely generate a response from a single prompt but operates as an agent capable of planning, making intermediate decisions, and interacting with external tools (e.g., search engines, databases, APIs, or computational modules). In such a configuration, the model can decompose a complex task into sub-goals, iteratively invoke the retrieval module, evaluate the obtained results, and update its internal state or operational context. The integration between RAG and agentic systems thus allows transitioning from static generation, limited to a single cycle of retrieval and response, to iterative and goal-oriented processes, better suited to complex and dynamic application scenarios.

Despite advancements in defining and implementing efficient and comprehensive RAG systems, several challenges remain — among others — the following:
- alignment between retrieval and generation  
- efficient knowledge selection  
- interpretability of systems  
- realistic performance evaluation  
- management of obsolete or contradictory information  
- definition of realistic benchmarks and adequate evaluation metrics

=== 1.5 Conclusion

The RAG paradigm has evolved from a simple document retrieval system to a complex, modular, and knowledge-oriented ecosystem. The dynamic integration of external knowledge allows improving the reliability, factuality, and transparency of language models, making such systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations could lead to the development of more robust, interpretable, and updatable intelligent systems.

== 2. Methods

The documentation has been preprocessed and indexed. The embeddings were generated using Amazon Bedrock.

The database used is PostgreSQL with vector support.

== 3. Results

The system has demonstrated:
- good semantic retrieval
- contextualized responses
- reduced search times

== 4. Discussion

Possible developments:
- integration with healthcare systems
- improved observability
- multimodality

== 5. References
#bibliography("bibliography.bib", style: "ieee")