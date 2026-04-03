```typst
#import "template.typ": address, contacts, copyright, disclaimer, ottante-report, remark, smallsection, tableofcontents

#show: ottante-report.with(
  title: "Design of a Multi-Agent RAG Architecture to Support the AREU (112) Operations Center",
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

The integration of large language models (LLM) in critical sectors is often hindered by structural limitations such as "hallucinations" and the static nature of knowledge.
This work describes the design and implementation of a multi-agent RAG (Retrieval-Augmented Generation) architecture aimed at supporting operators of the AREU (112) operations center in the rapid consultation of complex operational procedures and medical protocols.

The system is orchestrated through LangGraph, which allows overcoming the linearity of traditional pipelines in favor of a computational graph where specialized autonomous agents collaborate for query decomposition, semantic retrieval, and result validation.
To ensure maximum accuracy in retrieval, a "query expansion" strategy based on "hypothetical queries" and a reranking system based on a sigmoid function were adopted to minimize informational noise.
The entire infrastructure is hosted on Amazon Bedrock, ensuring high standards of security and confidentiality in handling sensitive data from the Agency.

Tests conducted with the direct involvement of AREU personnel demonstrate that the integration of reasoning mechanisms for each node of the graph significantly increases the transparency and reliability of the system.
Although the tested models (Amazon Nova-2-Lite and Claude Sonnet) showed similar satisfaction scores, performance analysis highlighted latency as a critical factor for emergency operations, identifying document filtering as the main bottleneck.
In conclusion, the work demonstrates how a multi-agent approach can transform corporate documentation into an intelligent and secure dialogue, providing concrete decision-making support in high-stress operational situations.

= Introduction and Context

The integration of Artificial Intelligence in critical sectors represents one of the most ambitious challenges of the modern era. Although large language models (LLM) have demonstrated extraordinary capabilities in understanding and generating texts, they carry significant structural limitations, such as the tendency to generate incorrect information (so-called "hallucinations") and possessing static knowledge, limited to the time of their training.
In a field where the accuracy of information can make the difference between the success and failure of an intervention, these limitations are unacceptable.

== The Role of AREU and the Management of 112

This work is positioned at the heart of the Lombard rescue system: the Regional Emergency Agency (AREU), the entity that coordinates the Unique Emergency Number 112.
Operating in an emergency center means managing extremely high-pressure situations, where the speed and reliability of information are crucial.
Operators must have immediate access to a vast wealth of knowledge that includes operational instructions, internal procedures, and complex medical protocols, often contained in dense documents that are not always easy to consult rapidly during a call.

== The Need for an Intelligent Assistant

The goal of this project is to create a multi-agent architecture based on the RAG (Retrieval-Augmented Generation) paradigm, specifically designed to support the personnel of the center.
Unlike a normal AI, this system does not rely solely on its internal "memory," but acts as an expert librarian: upon receiving a question, it queries an external memory composed of AREU's official documents, extracts relevant information, and reprocesses it to provide a clear and verifiable answer.
This approach ensures that every suggestion provided by the agent is anchored to reliable and up-to-date sources, drastically reducing the risk of errors.
Moreover, the choice to use advanced infrastructures such as Amazon Bedrock is not only technical but strategic: it allows managing sensitive data while ensuring maximum control and confidentiality of documents, an indispensable requirement when dealing with rescue procedures and internal protocols.

== Operational Impact and Future Vision

The system does not arise in isolation, but is the result of a process that has seen the direct involvement of AREU operators, who have tested the relevance of the answers and the ease of use of the interface.
The idea behind it is to offer a decision support tool that can serve different professional figures: from field rescuers to healthcare operators, and even logistics managers.
In summary, the introduction of this technology in AREU aims to transform the way information is retrieved in emergencies, shifting from a manual and potentially slow search to an intelligent dialogue with corporate documentation, thus helping to make decisions faster, more informed, and safer for the public.

= State of the Art: Retrieval-Augmented Generation (RAG) Systems

== Genesis of the RAG Paradigm

In recent years, Large Language Models (LLM) have demonstrated high capabilities in learning factual and semantic knowledge from large amounts of textual data. However, such models present significant structural limitations, including the difficulty of continuous updating, poor interpretability, and the tendency to generate incorrect or unverifiable information, a phenomenon known as "generation of hallucination." These limitations arise from the parametric nature of the knowledge incorporated in the models: it is encoded in the weights as a result of a large-scale optimization process and remains substantially static.

From a mathematical perspective, such knowledge corresponds to a point (or a region) of minimum of the loss function in the parameter space; modifying it in a targeted way therefore requires a new training phase or re-optimization, since it is impossible to intervene locally on individual contents without altering the global balance of the learned parameters.

To address these issues, the Retrieval-Augmented Generation (RAG) paradigm was introduced by Lewis et al. in 2020 @lewis2020rag. In this approach, the parametric knowledge of the model is complemented by a non-parametric external memory, composed of collections of pre-indexed documents.

== Architecture and Components of a RAG System

A RAG system is based on three fundamental components:
- retrieval from external sources
- knowledge integration
- text generation

Access to external sources of knowledge occurs through semantic retrieval techniques, i.e., methods that do not limit themselves to simple lexical matching between keywords, but represent queries and documents as vectors in a continuous semantic space (embedding). In this vector space, the similarity between texts is calculated based on the distance or angle between their respective vectors (for example via cosine similarity), allowing for the retrieval of relevant documents even when they do not share exactly the same terms as the query, but express the conceptual content. In this way, the model can dynamically integrate updated or specialized information without having to modify its internal parameters. The tool typically used for this phase of a RAG system is the neural retriever, a system composed of one or more neural encoders (typically based on Transformer architectures) trained to map both queries and documents into dense vectors belonging to a shared semantic space. In the most common configuration, called bi-encoder, one encoder encodes the query and another encodes the documents (sometimes with shared weights). Each text is thus mapped into a vector $f(x) in RR^d$, where the dimension $d$ is fixed a priori.

The Transformer architecture, introduced by Vaswani et al. (2017) @vaswani2017attention, is based exclusively on attention mechanisms, completely eliminating recurrences and convolutions. The central component is the self-attention, which allows each element of a sequence to dynamically weigh all other elements, modeling long-range dependencies in a parallel manner. Given a set of queries $Q$, keys $K$, and values $V$, the scaled attention (scaled dot-product attention) is defined as:

$
  attention(Q, K, V) = softmax((Q K^TT) / sqrt(d_k)) V
$

where $d_k$ is the size of the keys. The normalization factor
$1/sqrt(d_k)$ stabilizes the gradient for large vector dimensions.

Instead of applying a single attention function with keys, values, and queries of size $d_upright("model")$, it is advantageous to project linearly
the queries, keys, and values $h$ times through different learned linear projections, respectively towards spaces of size $d_k$, $d_k$, and $d_v$. Multi-head attention allows the model to jointly pay attention to information coming from different representation subspaces in different positions. With a single attention head, the averaging operation tends to inhibit this capacity:

$
  mh(Q, K, V) = concat(head_1, ..., head_h) W^O
$

$
  upright("where") quad head_i = attention(Q W_i^Q, K W_i^K, V W_i^V)
$

and $W_i^Q, W_i^K, W_i^V$ and $W^O$ are learned parametric matrices.

Thanks to this architecture, the Transformer allows for entirely parallel processing of sequences and efficient modeling of global dependencies, laying the foundation for modern large language models.

The loss function is the mathematical criterion that guides the training of the retriever. It is generally of a contrastive type: given a set of relevant (positive) and non-relevant (negative) query-document pairs, the loss is defined to maximize the similarity between the vectors of positive pairs and minimize that of negative pairs. Formally, the parameters of the encoder are optimized so that the geometric structure of the vector space reflects a notion of semantic relevance. In this way, the learned function induces a metric in the embedding space that approximates the concept of relevance relative to the query.

The most relevant documents identified by the retriever are provided as context to the generative model, which uses them to produce the answer. This integration between semantic retrieval and textual generation allows enriching the parametric knowledge of the model with updatable external information, improving performance in knowledge-intensive tasks.

This paradigm has proven to increase the factuality, transparency, and updatability of systems based on LLM, while reducing the need for continuous retraining. Additionally, the ability to track information sources during generation represents a key element for trust and verifiability of responses @rag_survey2023.

== Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, characterized by increasing modularity and a tighter integration between retrieval and generation. Early implementations followed a linear pipeline, where a query was transformed into an embedding, used to retrieve documents, and subsequently provided to the generative model @rag_survey2023.

In this initial phase, retrieval mechanisms largely derived from traditional Information Retrieval: lexical approaches such as BM25 @okapi_trec1994 were progressively complemented and then replaced by dense neural models, based on vector representations shared between queries and documents. The adoption of semantic embeddings improved the robustness and generalization capability of systems.

As the paradigm matured, research introduced modular improvements aimed at optimizing the quality of the context provided to the generator. These include strategies for chunking documents (segmenting into coherent informative units), query expansion, dynamic context selection, re-ranking, and semantic filtering. Such techniques reflect the awareness that the notion of relevance in generative systems differs from that of traditional retrieval systems @survey2025.

A further evolution concerns the integration between retrieval and reasoning.
Modern architectures include multi-hop approaches, where the system performs multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This allows addressing complex tasks that require distributed inferences across multiple documents or heterogeneous sources.

In this perspective, a central aspect becomes the alignment between retriever and generator. Zhou & Chen (2025) propose an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative context differs from that of traditional retrieval systems @openrag2025. This approach highlights how the efficiency of retrieval can significantly impact overall performance, even in the presence of smaller generative models.

At the same time, recent studies emphasize the importance of managing the context window of LLM. Semantic compression techniques, adaptive document selection, and noise control aim to optimize the use of available context. In particular, Cuconasu et al. (2024) show that appropriate filtering and weighting of retrieved documents reduce noise and improve the effectiveness of the model @noise2024, while Cheng et al. (2025) provide a systematic overview of the most recent strategies for efficiently combining and selecting documents @survey2025.

A counterintuitive result that has emerged in literature concerns the role of noise in the retrieval process. Cuconasu et al. (2024) show that the controlled introduction of not perfectly relevant documents can improve the accuracy of the generative model, suggesting that a more diverse context favors the inferential process @noise2024. A possible theoretical conjecture supporting this result is that the presence of moderately heterogeneous information acts as a form of contextual regularization: the inclusion of documents not strictly overlapping with the query could reduce the risk of overfitting the model to partial or redundant evidence, favoring a more robust latent representation. Furthermore, a diversified context could widen the semantic space explored by the model during generation, increasing the likelihood of activating relevant but not immediately obvious conceptual connections.

Another hypothesis is that "controlled noise" improves attention calibration: the need to discriminate between strongly and weakly relevant signals could lead the model to weigh the available information more selectively, reinforcing the mechanisms of integration and comparison between sources. In this perspective, the observed positive effect would not derive from the noise itself, but from its function of stimulating selection and inferential composition within the context window. This result challenges traditional metrics for evaluating retrieval and opens new perspectives in the design of RAG systems.

Finally, there is a growing interest in integrating with multimodal systems, structured databases, and knowledge graphs, expanding the role of retrieval beyond simple access to textual documents and configuring RAG systems as hybrid infrastructures for accessing and composing knowledge.

== Agentic Frontiers, Evaluations, and Future Challenges

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching enables the application of RAG systems in real-world contexts, characterized by latency and cost constraints. Moreover, integration with agentic systems and automated workflows represents a rapidly growing area. By agentic system, we mean an architecture in which the language model is not limited to generating a response based on a single prompt, but operates as an agent capable of planning, making intermediate decisions, and interacting with external tools (e.g., search engines, databases, APIs, or calculation modules).

The cutting edge of research is shifting towards Multi-Agent Systems (MAS). In this paradigm, the complexity of the task is not entrusted to a single "generalist agent," but is distributed among a network of specialized agents that collaborate with each other. This decomposition of tasks allows for assigning distinct roles to different logical entities, such as:
- Research Agent: specialized in semantic query expansion (hyper queries) and accurate retrieval of chunks.
- Critical or Validator Agent: tasked with assessing the relevance of sources and the coherence of the response, triggering correction cycles if quality criteria are not met.
- Synthesizer Agent: focused on the final text generation, ensuring that it is anchored to sources and free of hallucinations.
This transition towards a multi-agent approach theoretically justifies the adoption of architectures based on computational graphs, where each node is not a simple function but an agent with its own area of expertise.
The MAS approach increases the modularity and interpretability of the system: since each agent must provide a "reasoning" for its actions, it becomes possible to accurately trace where an error may occur in the reasoning chain.

Despite progress in defining and implementing efficient and comprehensive RAG systems, several challenges remain, including:
- alignment between retrieval and generation
- efficient knowledge selection
- interpretability of systems
- realistic performance evaluation
- management of obsolete or contradictory information
- definition of realistic benchmarks and appropriate evaluation metrics

== Conclusion

The RAG paradigm has evolved from a simple document retrieval system to a complex, modular, knowledge-oriented ecosystem. The dynamic integration of external knowledge allows for improving the reliability, factuality, and transparency of language models, making such systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations could lead to the development of more robust, interpretable, and updatable intelligent systems.

= Methods

== System Architecture

=== Selection of Database and Models

For the implementation of the system, a targeted selection was made both of the database used for storing information and the models employed in the retrieval and generation phases.
In particular, PostgreSQL was chosen as the database management system, enriched with pgvector, an extension that allows for efficient storage and search of vector embeddings.
This solution allows integrating vector search functionalities directly within a relational database, simplifying the system's infrastructure and facilitating the management of documents and their embeddings used in the retrieval process.

Regarding artificial intelligence models, the Amazon Bedrock service was used, which provides access to various foundational models through a unified interface.
Using this platform allows for integrating models for embedding and generation, maintaining a flexible and scalable architecture while also facilitating experimentation and replacement of models without substantial modifications to the system's infrastructure.
Moreover, the use of Amazon Bedrock allows for managing access to models via the Amazon Web Services infrastructure, ensuring greater control over data and helping to preserve the confidentiality of documents used by the system, a particularly relevant aspect in the context of this work.

=== LangGraph

For the implementation and orchestration of the workflow of the RAG system, it was decided to use LangGraph, a library developed for building applications based on Large Language Models through graph-based computational structures.
The choice comes from the need to define a structured processing process, which involves dynamic integrations between different phases, modularity, and flexibility in execution, thus ensuring the robustness required in a critical context like that of AREU.
LangGraph extends the paradigm of linear pipelines typically used in LLM frameworks, allowing for defining more complex execution flows characterized by conditional transitions and iterative cycles.

Unlike traditional sequential architectures, where operations are executed according to a static pipeline, LangGraph represents the process as a directed graph composed of nodes and edges.
In this context, each node represents a computational unit — for example, a retrieval, reasoning, or generation module — while the edges define the flow of data and the transition conditions between the different phases of the system.
In practice, each node is an autonomous agent executing a specific function.

The processing is also guided by a shared "state" of the graph (graph state), which is a data structure containing the relevant information accumulated during execution.
This state is passed between agents and progressively updated, allowing the system to keep track of the current context of the request and the intermediate results produced by the different components of the workflow.

The use of a graph-based architecture has enabled the implementation of more articulated reasoning strategies, such as iterations between retrieval and generation, quality checks on produced responses, or fallback mechanisms when retrieved information proves insufficient.
At the same time, this structure ensures high modularity: each node of the graph encapsulates a specific functionality of the system, allowing for modifying or replacing individual components — such as the retriever or the generative model — without altering the overall workflow.
These features make the system more flexible, facilitating both dynamic adaptation during execution and experimentation with different architectural configurations.

=== Pipeline Description

The workflow of the RAG architecture is represented in the image @fig:workflow, which illustrates all phases of the process.
Following the receipt of a query, an agent decides whether to end the conversation, generate a direct response without retrieving documents, or proceed with retrieval. This strategy allows avoiding the transition from the retrieval phase for questions that do not require the use of documents, thus optimizing both computational costs and response times.

In case of retrieval, another agent generates two questions semantically close to the original one (hypothetical queries), embeds all three queries, and uses the average of the three embeddings to retrieve the most relevant documents from the database.
The retrieved documents are then provided as context to the generative agent, which produces the final response.

The next step consists in validating the generated response, which is compared with the original question to verify the coherence and correctness of the information provided. In case of a negative outcome, the system may decide to iterate again the retrieval process, updating the graph state with the information obtained and generating new queries to further refine the search, or to terminate the conversation if it is deemed that further attempts will not lead to a significant improvement in the response (for example, due to insufficient documentation).

If the response is deemed satisfactory, the system returns to the end user a contextualized answer based on external sources, thus improving the reliability and relevance of the information provided compared to a standalone generative model.

Everything generated during the process (queries, retrieved documents, produced responses, time taken by each node, ...) is stored in the state of the graph, allowing the system to maintain a complete trace of the interaction and to use this information for any subsequent iterations or for post-hoc analyses of the system's performance.
Moreover, all steps are accompanied by "reasoning," i.e., a textual explanation that describes the decisions made by the system at each phase, thus improving the transparency and interpretability of the process.

#figure(
  image("/ObservableRagAgentTest-graph.png", width: 100%),
  caption: [RAG agent workflow],
) <fig:workflow>

== Dataset and Preprocessing

The documents used for retrieval were collected from public and private sources and include operational instructions, internal procedures, and relevant documents for emergency management.
The preprocessing process involved data cleaning, text normalization, and segmentation into coherent informative units (chunking). Each sufficiently long "chunk" is accompanied by a "summary" and three "hypothetical queries," i.e., automatically generated questions whose answers are contained within the chunk and represent different informational aspects, in order to improve semantic coverage during the retrieval process. It has been observed that having "hypothetical queries" alongside the textual body of the chunk leads to a significant improvement in the retrieval phase.
All chunks have been enriched with keywords taken from the general context of the document, so as to also preserve relevant information that, while distributed at different points in the text, is fundamental for the correct understanding of the content, ensuring that it is not lost.

The database was implemented using PostgreSQL with the pgvector extension, which allows for efficiently storing and indexing the vector embeddings associated with the documents.
The embeddings were generated using Titan Text Embeddings V2, an embedding model available on Amazon Bedrock, and stored in the database along with the document metadata, such as title and distinction between public and private documents.

== Retrieval with Reasoning

In the retrieval phase, the system generates two hypothetical queries based on the user's question, transforms them into embeddings, and uses the average of the three embeddings to retrieve the most relevant documents from the database. Considering the documents closest to the "semantic mean" of three similar questions allows for more accurately identifying the chunks that are truly useful for generating the response. Subsequently, an evaluation of the content of the retrieved documents is made, aiming to identify the relevance and pertinence of the information relative to the original query. This evaluation is performed by a model, which assigns to each "chunk" a score between 0 and 1, representing the probability that the document is relevant to the query, along with a textual explanation describing the reasons for the evaluation.

After this phase, the documents are ranked based on relevance and filtered to retain only the most pertinent ones through a "sigmoid" centered at the average of the scores with steepness 8, in order to reduce noise and improve the quality of the context provided to the generative model. It is not the raw relevance that determines the filter: it is first transformed via the sigmoid.
Consequently, only the documents for which the sigmoid value is greater than 0.65 are retained, while those for which the sigmoid returns a value ≤ 0.65 are discarded, as shown in @fig:sigmoid. A "re-ranking" and "filtering" solution of this type allows for discarding non-relevant chunks, without the risk of losing useful information.

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

Finally, the selected documents are provided as context to the generative model, maintaining information about relevance to the original query, to enable the agent to integrate information more effectively during the generation of the response.

== Generation and Validation

The generative model uses as context the retrieved documents with the relevance score, the original query, and the corresponding hypothetical queries, to produce a contextualized and source-based response. The generation of the response is guided by a Jinja2 prompt, designed to encourage the model to integrate information coherently and to provide explanations for the decisions made during generation. The use of Jinja2 also allows for dynamically structuring the prompt, facilitating the insertion and management of variables (such as documents and queries) in a modular, reusable, and easily maintainable way. The response contains citations to the documents used, along with a summary of the content and an explanation of the inferential process followed by the model to arrive at the final answer.

The validation phase directs the flow of the system towards iterating the retrieval or towards terminating the conversation. Even in this step, the agent provides a textual explanation that describes the reasons behind the decision made, thus improving the transparency of the process and allowing the user to understand why the system deemed the response satisfactory or not. Specifically, the model assigns a satisfaction score between 0 and 1, calculated based on defined criteria; a response is considered acceptable when this score is equal to or greater than 0.8.

= Results

Preliminary tests were conducted to evaluate the effectiveness of the implemented RAG system, focusing on the quality of semantic retrieval, quality of generated responses, and search times. The results obtained indicate that the system is capable of retrieving relevant documents even in the presence of complex or ambiguous queries, thanks to the use of semantic embeddings and the generation of hypothetical queries. The produced responses are contextualized and based on reliable sources, with a significant reduction in search times compared to a purely manual approach.

The most impactful component on the quality of retrieval was the presence of hypothetical queries: almost always, it was sufficient to add the right question to correctly retrieve the most significant chunk. Secondly, a well-structured user input allows for a qualitative leap in this phase. Regarding the response generation phase, particular attention was dedicated to prompt engineering, as it emerged that the prompt represents a determining factor for the quality of the output. Given the same informative context and well-defined query, the structure, clarity, and instructions contained in the prompt significantly influence the coherence, completeness, and reliability of the generated response. Moreover, a clear and precise question enables the model to understand more accurately which information to include in the response, reducing ambiguity and improving the relevance of the result.

AREU operators were involved in conducting the tests, providing qualitative feedback on the relevance of responses and the ease of use of the system. To automate the testing process and have a benchmark for evaluating future developments and improvements, a dataset of questions accompanied by the expected content that must be present in their respective answers was built. Preliminary results suggest significant potential for the application of RAG systems in emergency contexts, where the speed and reliability of information are crucial.

Two different generative models were compared to evaluate the impact of model quality on the overall performance of the RAG system: Amazon Nova 2 Lite and Claude Sonnet 4.6. As shown in @fig:boxplot_scores, no significant differences were found between the two models in terms of satisfaction score assigned to the generated responses, suggesting that the quality of retrieval and the structure of the prompt may have a more decisive impact on overall performance than the specific choice of generative model, at least within the range of tested models. This result highlights the importance of optimizing retrieval and prompt engineering components to maximize the effectiveness of RAG systems, rather than focusing solely on the selection of the generative model. Regarding response times, from @fig:boxplot_times a significant difference in latency is observed between the two models, with the Amazon Nova 2 Lite model returning responses in significantly shorter times than the other. This result underscores the importance of considering not only the quality of generated responses but also the computational efficiency of the models used, especially in contexts like that of AREU, where the speed of information is crucial.

Looking more closely at the times in the various nodes of the graph (@fig:nodes_times_comparison), it was found that the re-ranking phase represents the main bottleneck, with execution times significantly higher than in the generation phase. This result highlights the importance of optimizing the evaluation and selection phase of retrieved documents, as it represents a critical point for the overall efficiency of the RAG system. Optimizing this phase could lead to significant improvements in response times, making the system more suitable for real-world scenarios where the speed of information is essential.

#figure(
  image("/boxplot_score.png", width: 100%),
  caption: [Boxplot scores],
) <fig:boxplot_scores>

#figure(
  image("/boxplot_times.png", width: 100%),
  caption: [Boxplot times],
) <fig:boxplot_times>

#figure(
  image("/nodes_times_comparison.png", width: 100%),
  caption: [nodes times comparison],
) <fig:nodes_times_comparison>

= Discussion

The implemented RAG system has proven effective in retrieving relevant information and generating contextualized responses, improving the reliability and relevance of the information provided to AREU operators. However, some limitations have emerged, including the need for further optimizations to manage more complex or dynamic scenarios. Specifically, the documents used for retrieval were not produced with the goal of being processed by a RAG system, and therefore present non-optimal structures for the retrieval and generation process.
In particular, the presence of redundant information (the same concepts repeated in different sections of the same document or distributed across different documents), obsolete (outdated directives), or unstructured (images and tables) made it more challenging for the system to identify and integrate the most relevant information, highlighting the importance of a more targeted data curation and organization process to maximize the effectiveness of RAG systems.
Having documents designed from the outset for a RAG system allows for reducing the number of chunks to be sent to the re-ranking node, thus optimizing overall performance, as the system takes less time both to process them and to reason about the contents.

The strengths of the system include the modularity of the architecture (a strong point of a multi-agent architecture), which allows for replacing or updating individual components without altering the entire workflow, and the ability to dynamically integrate external information, improving the factuality and transparency of the generated responses. Furthermore, the use of a graph-based approach has allowed for implementing more articulated reasoning strategies and maintaining a complete trace of the interaction, facilitating subsequent iterations and post-hoc analyses of the system's performance.

Among future developments, it is planned to explore integration with existing healthcare systems, in order to further improve the relevance and utility of the responses provided to AREU operators. Another research direction concerns the exploration of multimodality, integrating non-textual knowledge sources such as images or structured data to further enrich the context provided to the generative model. Finally, it is planned to conduct a systematic comparison of different embedding and generation models in order to identify the most effective configurations for the specific context of AREU, and to explore hierarchical ranking strategies to further optimize document selection during the retrieval process.

In conclusion, a RAG system proves particularly valuable in emergency situations, as it enhances operational efficiency, ensuring that responses are always based on reliable documents. The flexibility and adaptability of the system make it usable in multiple scenarios, supporting various professionals, from rescuers to healthcare operators, and even logistics and communication managers, thus contributing to faster and more informed decisions.

#bibliography("bibliography.bib", style: "ieee")
```