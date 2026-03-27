#import "template.typ": address, contacts, copyright, disclaimer, ottante-report, remark, smallsection, tableofcontents

#show: ottante-report.with(
  title: "Design of a RAG Agent for Support to the AREU (112) Central",
  subtitle: "",
  authors: "Andrea De Tomasi",
  date: "April 2026",
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

= Introduction

---

= State of the Art: Retrieval-Augmented Generation (RAG) Systems

== Genesis of the RAG Paradigm

In recent years, Large Language Models (LLMs) have demonstrated high capabilities in learning factual and semantic knowledge from large amounts of textual data. However, these models exhibit significant structural limitations, including difficulties in continuous updating, poor interpretability, and a tendency to generate incorrect or unverifiable information, a phenomenon known as "generation of hallucination." These limitations arise from the parametric nature of the knowledge embedded in the models: it is encoded in the weights as a result of a large-scale optimization process and remains substantially static.

From a mathematical standpoint, such knowledge corresponds to a point (or region) of minimum in the loss function in the parameter space; thus, targeted modification requires a new training phase or re-optimization, as it is not possible to intervene locally on individual contents without altering the global balance of the learned parameters.

To address these issues, the Retrieval-Augmented Generation (RAG) paradigm was introduced by Lewis et al. in 2020 @lewis2020rag. In this approach, the parametric knowledge of the model is complemented by an external non-parametric memory, composed of collections of pre-indexed documents.

== Architecture and Components of a RAG System

A RAG system is based on three fundamental components:
- retrieval from external sources
- knowledge integration
- text generation

Access to external knowledge sources occurs through semantic retrieval techniques, which are methods that go beyond simple lexical matching of keywords, representing queries and documents as vectors in a continuous semantic space (embedding). In this vector space, the similarity between texts is calculated based on the distance or angle between their respective vectors (for example, via cosine similarity), allowing for the retrieval of relevant documents even when they do not share exactly the same terms as the query, but express its conceptual content. This way, the model can dynamically integrate updated or specialized information without having to modify its internal parameters. The tool typically used for this phase of a RAG system is the neural retriever, a system composed of one or more neural encoders (typically based on Transformer architectures) trained to map both queries and documents into dense vectors belonging to a shared semantic space. In the most common configuration, called bi-encoder, one encoder encodes the query and another encodes the documents (sometimes with shared weights). Each text is thus mapped to a vector $f(x) in RR^d$, where the dimension $d$ is fixed a priori.

The Transformer architecture, introduced by Vaswani et al. (2017) @vaswani2017attention, relies exclusively on attention mechanisms, completely eliminating recurrence and convolutions. The central component is self-attention, which allows each element of a sequence to dynamically weigh all other elements, modeling long-range dependencies in parallel. Given a set of queries $Q$, keys $K$, and values $V$, the scaled attention (scaled dot-product attention) is defined as:

$
  attention(Q, K, V) = softmax((Q K^TT) / sqrt(d_k)) V
$

where $d_k$ is the dimension of the keys. The normalization factor
$1/sqrt(d_k)$ stabilizes the gradient for large vector sizes.

Instead of applying a single attention function with keys, values, and queries of size $d_upright("model")$, it is beneficial to linearly project the queries, keys, and values $h$ times through different learned linear projections, respectively into spaces of size $d_k$, $d_k$, and $d_v$. Multi-head attention allows the model to pay attention jointly to information coming from different representation subspaces at different positions. With a single head of attention, the averaging operation tends to inhibit this capability:

$
  mh(Q, K, V) = concat(head_1, ..., head_h) W^O
$

$
  upright("where") quad head_i = attention(Q W_i^Q, K W_i^K, V W_i^V)
$

and $W_i^Q, W_i^K, W_i^V$ and $W^O$ are learned parametric matrices.

Thanks to this architecture, the Transformer allows for fully parallel processing of sequences and efficient modeling of global dependencies, laying the foundation for modern large language models.

The loss function is the mathematical criterion that guides the training of the retriever. It is generally of a contrastive type: given a set of relevant (positive) and non-relevant (negative) query-document pairs, the loss is defined to maximize the similarity between the vectors of positive pairs and minimize that of negative pairs. Formally, the parameters of the encoder are optimized so that the geometric structure of the vector space reflects a notion of semantic relevance. In this way, the learned function induces a metric in the embedding space that approximates the concept of relevance to the query.

The most relevant documents identified by the retriever are provided as context to the generative model, which uses them to produce the answer. This integration between semantic retrieval and text generation allows for enriching the parametric knowledge of the model with updatable external information, improving performance on knowledge-intensive tasks.

This paradigm has proven to increase the factuality, transparency, and updatability of LLM-based systems, while reducing the need for continuous retraining. Furthermore, the ability to track information sources during generation represents a key element for trust and verifiability of responses @rag_survey2023.

== Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, characterized by increasing modularity and a tighter integration between retrieval and generation. Early implementations followed a linear pipeline, where a query was transformed into an embedding, used to retrieve documents, and subsequently provided to the generative model @rag_survey2023.

In this initial phase, retrieval mechanisms largely derived from traditional Information Retrieval: lexical approaches such as BM25 @okapi_trec1994 were progressively complemented and then replaced by dense neural models, based on vector representations shared between queries and documents. The adoption of semantic embeddings improved the robustness and generalization capacity of systems.

With the maturation of the paradigm, research has introduced modular improvements aimed at optimizing the quality of the context provided to the generator. These include strategies for chunking documents (segmenting into coherent informative units), query expansion, dynamic context selection, re-ranking, and semantic filtering. Such techniques reflect the awareness that the notion of relevance in generative systems differs from that of traditional retrieval systems @survey2025.

A further evolution concerns the integration between retrieval and reasoning. Modern architectures include multi-hop approaches, where the system performs multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This allows for tackling complex tasks that require distributed inferences across multiple documents or heterogeneous sources.

In this perspective, a central aspect becomes the alignment between the retriever and the generator. Zhou & Chen (2025) propose an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative context differs from that of traditional retrieval systems @openrag2025. This approach highlights how the efficiency of retrieval can significantly impact overall performance, even in the presence of smaller generative models.

At the same time, recent studies emphasize the importance of managing the LLM context window. Techniques for semantic compression, adaptive document selection, and noise control aim to optimize the use of available context. In particular, Cuconasu et al. (2024) show that appropriate filtering and weighting of retrieved documents reduce noise and improve model effectiveness @noise2024, while Cheng et al. (2025) provide a systematic overview of the latest strategies for efficiently combining and selecting documents @survey2025.

An intuitive result that has emerged in the literature concerns the role of noise in the retrieval process. Cuconasu et al. (2024) show that the controlled introduction of documents that are not perfectly relevant can improve the accuracy of the generative model, suggesting that a more diversified context favors the inferential process @noise2024. A possible theoretical conjecture supporting this result is that the presence of moderately heterogeneous information acts as a form of contextual regularization: the inclusion of documents that are not strictly overlapping with the query might reduce the risk of overfitting the model to partial or redundant evidence, favoring a more robust latent representation. Moreover, a diversified context could expand the semantic space explored by the model during generation, increasing the likelihood of activating relevant but not immediately obvious conceptual connections.

Another hypothesis is that "controlled noise" improves attention calibration: the need to discriminate between strongly and weakly relevant signals could lead the model to weigh available information more selectively, strengthening the mechanisms of integration and comparison between sources. In this perspective, the observed positive effect would not stem from the noise itself but from its function as a stimulus for selection and inferential composition within the context window. This result questions traditional evaluation metrics for retrieval and opens new perspectives in the design of RAG systems.

Finally, there is a growing interest in integration with multimodal systems, structured databases, and knowledge graphs, expanding the role of retrieval beyond simple access to textual documents and configuring RAG systems as hybrid infrastructures for accessing and composing knowledge.

== Agent Frontiers, Evaluations, and Future Challenges

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching allows RAG systems to be applied in real-world contexts characterized by latency and cost constraints. Furthermore, integration with agent-based systems and automated workflows represents a rapidly growing area. An agent-based system refers to an architecture where the language model does not simply generate a response based on a single prompt but operates as an agent capable of planning, making intermediate decisions, and interacting with external tools (e.g., search engines, databases, APIs, or computing modules). In such a configuration, the model can decompose a complex task into sub-goals, iteratively call the retrieval module, evaluate the obtained results, and update its internal state or operational context. The integration of RAG and agent-based systems thus enables a shift from static generation, limited to a single cycle of retrieval and response, to iterative and goal-oriented processes better suited to complex and dynamic application scenarios.

Despite advances in the definition and implementation of efficient and comprehensive RAG systems, several challenges remain — among others:
- alignment between retrieval and generation
- efficient knowledge selection
- interpretability of systems
- realistic performance evaluation
- management of obsolete or contradictory information
- definition of realistic benchmarks and appropriate evaluation metrics

== Conclusion

The RAG paradigm has evolved from a simple document retrieval system to a complex, modular, and knowledge-oriented ecosystem. The dynamic integration of external knowledge allows for improving the reliability, factuality, and transparency of language models, making these systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations could lead to the development of more robust, interpretable, and updatable intelligent systems.

= Methods

== System Architecture

=== Database and Model Selection

For the implementation of the system, a targeted selection was made of both the database used for storing information and the models employed in the retrieval and generation phases. In particular, PostgreSQL was chosen as the database management system, extended via pgvector, an extension that allows for efficient storage and retrieval of vector embeddings. This solution integrates vector search functionality directly within a relational database, simplifying the system's infrastructure and facilitating the management of documents and their corresponding embeddings used in the retrieval process.

Regarding artificial intelligence models, the Amazon Bedrock platform was used, providing access to various foundational models through a unified interface. The use of this platform allows for integrating models for embedding and generation while maintaining a flexible and scalable architecture, simultaneously facilitating experimentation and replacement of models without substantial changes to the system's infrastructure. Additionally, the use of Amazon Bedrock allows for managing access to models through the Amazon Web Services infrastructure, ensuring greater control over the data and helping to preserve the confidentiality of the documents used by the system, an aspect particularly relevant in the context of this work.

=== LangGraph

For the implementation and orchestration of the RAG system workflow, LangGraph was used, a library developed for building applications based on Large Language Models through graph-based computational structures. LangGraph extends the paradigm of linear pipelines typically used in LLM frameworks, allowing for defining more complex execution flows characterized by conditional transitions and iterative loops.

Unlike traditional sequential architectures, where operations are performed according to a static pipeline, LangGraph represents the process as a directed graph composed of nodes and edges. In this context, each node represents a computational unit — for example, a retrieval, reasoning, or generation module — while the edges define the flow of data and transition conditions between the different phases of the system.

The processing is also guided by a shared "graph state," which is a data structure containing the relevant information accumulated during execution. This state is passed between nodes and progressively updated, allowing the system to keep track of the current context of the request and the intermediate results produced by the different components of the workflow.

Using a graph-based architecture allows for implementing more elaborate reasoning strategies, such as iterations between retrieval and generation, quality checks on the produced responses, or fallback mechanisms when retrieved information is insufficient. At the same time, this structure ensures high modularity: each node of the graph encapsulates a specific functionality of the system, allowing for modifying or replacing individual components — such as the retriever or the generative model — without altering the overall workflow. These features make the system more flexible, facilitating both dynamic adaptation during execution and experimentation with different architectural configurations.

=== Pipeline Description

The workflow of the RAG system is represented in the image @fig:workflow, which illustrates all the phases of the process. Following the receipt of a query, the system decides whether to terminate the conversation, generate a direct response without retrieving documents, or proceed with retrieval. This strategy allows for avoiding the retrieval phase for questions that do not require the use of documents, thus optimizing both computational costs and response times.

In the case of retrieval, the agent generates two semantically close questions to the original one (hyper queries), embeds all three queries, and uses the average of the three embeddings to retrieve the most relevant documents from the database. The retrieved documents are then provided as context to the generative model, which produces the final response.

The next step consists of validating the generated response, which is compared with the original question to verify the consistency and correctness of the information provided. In the case of a negative outcome, the system may decide to iterate the retrieval process again, updating the graph state with the obtained information and generating new queries to further refine the search, or to terminate the conversation if it is believed that further attempts will not lead to a significant improvement in the response (for example, due to insufficient documentation).

If the response is deemed satisfactory, the system terminates the conversation, returning to the end user a contextualized answer based on external sources, thereby improving the reliability and relevance of the information provided compared to a standalone generative model.

Everything generated during the process, including queries, retrieved documents, produced responses, and the time taken by each node, is stored in the graph state, allowing the system to maintain a complete record of the interaction and to use this information for any subsequent iterations or for post-hoc analysis of the system's performance. Additionally, all steps are accompanied by "reasoning," meaning a textual explanation that describes the decisions made by the system at each stage, thereby improving the transparency and interpretability of the process.

#v(2.0em)
#figure(
  image("/ObservableRagAgentTest-graph.png", width: 100%),
  caption: [RAG agent workflow],
) <fig:workflow>

#v(2.0em)

== Dataset and Preprocessing

The documents used for retrieval were collected from public and private sources and include operational instructions, internal procedures, and documents relevant to emergency management. The preprocessing process involved data cleaning, text normalization, and segmentation into coherent informative units (chunking). Each sufficiently long "chunk" is accompanied by a "summary" and three "hyper queries," which are automatically generated questions that represent various aspects of the chunk's content, in order to improve semantic coverage during the retrieval process. It has been observed that having "hyper queries" alongside the textual body of the chunk leads to a significant improvement in the retrieval phase.

The database was implemented using PostgreSQL with the pgvector extension, which allows for efficiently storing and indexing the vector embeddings associated with the documents. The embeddings were generated using Titan Text Embeddings V2, an embedding model available on Amazon Bedrock, and stored in the database alongside the document metadata, such as title and distinction between public and private documents.

== Retrieval with Reasoning

In the retrieval phase, the system generates two hyper queries from the original query, transforms them into embeddings, and uses the average of the three embeddings to retrieve the most relevant documents from the database. Considering the documents closest to the "semantic mean" of three similar questions allows for more accurately identifying the chunks that are genuinely useful for generating the response. Subsequently, an evaluation of the content of the retrieved documents is made, with the aim of identifying the relevance and pertinence of the information with respect to the original query. This evaluation is performed by a model, which assigns to each "chunk" a score between 0 and 1, representing the probability that the document is relevant to the query, along with a textual explanation describing the reasons for the evaluation.

After this phase, the documents are ordered based on relevance and filtered to retain only the most pertinent ones through a "sigmoid" centered on the mean of the scores with steepness 8, in order to reduce noise and improve the quality of the context provided to the generative model. It is not the raw relevance that determines the filter: it is first transformed via the sigmoid. Consequently, only documents for which the sigmoid value is greater than 0.65 are retained, while those for which the sigmoid returns a value ≤ 0.65 are discarded, as shown in @fig:sigmoid. A "re-ranking" and "filtering" solution of this kind allows for rejecting non-relevant chunks without running the risk of losing useful information.

#let sample-relevances = (0.82, 0.91, 0.45, 0.78, 0.23, 0.67, 0.55, 0.38, 0.71, 0.60)

#let mean-relevance = sample-relevances.sum() / sample-relevances.len()
#let logit(y) = calc.log(y / (1 - y))
#let x-cut(mean, k, y-threshold) = {
  mean + (1 / k) * logit(y-threshold)
}
#let cutoff = x-cut(mean-relevance, 8, 0.65)

#figure(
  canvas({
    plot.plot(
      size: (10, 6),
      x-label: [relevance score],
      y-label: [$sigma$(relevance)],
      x-min: 0,
      x-max: 1,
      y-min: 0,
      y-max: 1,
      x-tick-step: 0.25,
      y-tick-step: 0.25,
      {
        // rejected zone
        plot.add-fill-between(
          domain: (0, cutoff),
          samples: 100,
          x => sigmoid(x, k: 8, x0: mean-relevance),
          x => 0,
          style: (fill: red.lighten(70%), stroke: none),
        )

        // accepted zone
        plot.add-fill-between(
          domain: (cutoff, 1),
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
      },
    )

    // vertical dashed line at the mean
    let mean-x = cutoff * 10 // maps [0,1] -> [0, size.x]
    draw.line(
      (mean-x, 0),
      (mean-x, 6),
      stroke: (paint: blue, dash: "dashed"),
    )

    // label the mean
    draw.content(
      (mean-x + 0.15, 5.5),
      text(size: 8pt, fill: blue)[cutoff = #calc.round(cutoff, digits: 2)],
    )
  }),
  caption: [Sigmoid],
) <fig:sigmoid>

Finally, the selected documents are provided as context to the generative model, maintaining information about their relevance to the original query, allowing the agent to more effectively integrate information during the generation of the response.

== Generation and Validation

The generative model uses the retrieved documents with the relevance score, the original query, and the related hyper queries as context to produce a contextualized response based on reliable sources. The generation of the response is guided by a Jinja2 prompt, designed to encourage the model to integrate information coherently and to provide explanations for the decisions made during generation. The use of Jinja2 also allows for dynamically structuring the prompt, facilitating the insertion and management of variables (such as documents and queries) in a modular, reusable, and easily maintainable way. The response contains citations to the documents used, along with a summary of the content and an explanation of the inferential process followed by the model to arrive at the final answer.

The validation phase directs the flow of the system towards either iterating retrieval or terminating the conversation. Even in this step, the agent provides a textual explanation describing the reasons behind the decision made, thus improving the transparency of the process and allowing the user to understand why the system deemed the response satisfactory or not. Specifically, the model assigns a satisfaction score between 0 and 1, calculated based on defined criteria; a response is considered acceptable when this score is equal to or greater than 0.8.

= Results

Preliminary tests were conducted to assess the effectiveness of the implemented RAG system, focusing on the quality of semantic retrieval, quality of generated responses, and search times. The results obtained indicate that the system is capable of retrieving relevant documents even in the presence of complex or ambiguous queries, thanks to the use of semantic embeddings and the generation of hyper queries. The produced responses are contextualized and based on reliable sources, with a significant reduction in search times compared to an approach based solely on generative models without retrieval.

The most impactful component on the quality of retrieval was the presence of hyper queries: almost always, it sufficed to add the right question to correctly retrieve the necessary chunk to answer the query. Secondly, a well-structured input from the user allows for a significant leap in this phase. Regarding the response generation phase, particular attention was dedicated to prompt engineering, as it emerged that the prompt is a determining factor for the quality of the output. Given an equally informative context and a well-defined query, the structure, clarity, and instructions contained in the prompt significantly influence the coherence, completeness, and reliability of the generated response. Moreover, a clear and precise question allows the model to understand more accurately which information to include in the response, reducing ambiguity and improving the relevance of the result.

For the execution of the tests, AREU operators were involved, who provided qualitative feedback on the relevance of the responses and the ease of use of the system. To automate the testing process and establish a benchmark for evaluating future developments and improvements, a dataset of questions was constructed, accompanied by the expected content that should be present in their respective responses. Preliminary results suggest significant potential for applying RAG systems in emergency contexts, where the speed and reliability of information are crucial.

Two different generative models were compared to assess the impact of model quality on the overall performance of the RAG system. The results indicate that, although both models are capable of effectively integrating retrieved information, the model with more advanced generative capabilities produces more coherent and detailed responses, highlighting the importance of a high-quality generative model to maximize the benefits of the RAG paradigm. (TODO performance and accuracy graphs)

= Discussion

The implemented RAG system has proven effective in retrieving relevant information and generating contextualized responses, improving the reliability and relevance of the information provided to AREU operators. However, some limitations have emerged, including dependence on the quality of the database and the models used, as well as the need for further optimizations to handle more complex or dynamic scenarios. Specifically, the documents used for retrieval were not produced with the aim of being processed by a RAG system, and thus present a non-optimal structure for the retrieval and generation process. In particular, the presence of redundant, obsolete, or unstructured information made it more difficult for the system to identify and integrate the most relevant information, highlighting the importance of a more targeted data curation and organization process to maximize the effectiveness of RAG systems.

The strengths of the system include the modularity of the architecture, which allows for replacing or updating individual components without altering the entire workflow, and the ability to dynamically integrate external information, improving the factuality and transparency of the generated responses. Furthermore, the use of a graph-based approach has allowed for implementing more elaborate reasoning strategies and maintaining a complete record of the interaction, facilitating subsequent iterations and post-hoc analyses of the system's performance.

Among future developments, it is anticipated to explore integration with existing healthcare systems to further improve the relevance and utility of the responses provided to AREU operators. Another research direction involves exploring multimodality, integrating non-textual knowledge sources such as images or structured data to further enrich the context provided to the generative model. Finally, there is a plan to conduct a systematic comparison between different embedding and generation models, in order to identify the most effective configurations for the specific context of AREU, and to explore hierarchical ranking strategies to further optimize document selection during the retrieval process.

In conclusion, a RAG system proves particularly valuable in emergency situations, as it enhances operational efficiency, ensuring that responses are always based on reliable documents. The flexibility and adaptability of the system make it usable in multiple scenarios, supporting various professionals, from rescuers to healthcare operators, and logistics and communications managers, thereby contributing to faster and more informed decisions. (TODO more precision on use cases and professionals)

#bibliography("bibliography.bib", style: "ieee")