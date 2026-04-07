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
This work describes the design and implementation of a multi-agent RAG (Retrieval-Augmented Generation) architecture intended to support AREU (112) operators in the rapid consultation of complex procedures and medical protocols.

The system is orchestrated using LangGraph, enabling it to overcome the linearity of traditional pipelines by leveraging a computational graph in which specialized autonomous agents collaborate on query decomposition, semantic retrieval, and result validation.
To maximize retrieval accuracy, a query expansion strategy based on hypothetical queries was implemented, along with a reranking system using a sigmoid function to minimize informational noise.
The entire infrastructure is hosted on Amazon Bedrock, ensuring high standards of security and confidentiality when handling sensitive data from the Agency..

Tests conducted with the direct involvement of AREU personnel demonstrate that integrating reasoning mechanisms at each node of the graph significantly enhances the system’s transparency and reliability.
Although the evaluated models (Amazon Nova-2 Lite and Claude Sonnet) achieved similar satisfaction scores, performance analysis identified latency as a critical factor for emergency operations, with document filtering emerging as the primary bottleneck.
In conclusion, this work illustrates how a multi-agent approach can transform corporate documentation into an intelligent and secure dialogue, providing tangible decision-making support in high-stress operational scenarios.

= Introduction and Context

The integration of Artificial Intelligence in critical sectors represents one of the most ambitious challenges of the modern era. While large language models (LLMs) have demonstrated remarkable capabilities in understanding and generating text, they exhibit significant structural limitations, including a tendency to produce incorrect information (so-called “hallucinations”) and possessing static knowledge, limited to the period of their training.
In domains where information accuracy can determine the success or failure of an intervention, these limitations are unacceptable.

== The Role of AREU and the Management of 112

This work is situated at the core of the Lombard rescue system: the Regional Emergency Agency (AREU), the organization responsible for coordinating the Unique Emergency Number 112.
Working in an emergency center entails managing extremely high-pressure situations, where both the speed and reliability of information are critical.
Operators must have immediate access to a vast body of knowledge, including operational instructions, internal procedures, and complex medical protocols—often contained in dense documents that are not always easy to consult quickly during a call.

== The Need for an Intelligent Assistant

The goal of this project is to develop a multi-agent architecture based on the RAG (Retrieval-Augmented Generation) paradigm, specifically designed to support the center’s personnel.
Unlike a conventional AI system, this architecture does not rely solely on its internal “memory”; instead, it functions as an expert librarian: upon receiving a query, it consults an external memory composed of AREU’s official documents, extracts relevant information, and processes it to provide a clear and verifiable response.
This approach ensures that every suggestion provided by the agent is grounded in reliable and up-to-date sources, significantly reducing the risk of errors.
Furthermore, the decision to leverage advanced infrastructures such as Amazon Bedrock is both technical and strategic, enabling the management of sensitive data while maintaining maximum control and confidentiality—an essential requirement when handling rescue procedures and internal protocols.

== Operational Impact and Future Vision

The system did not emerge in isolation but is the result of a process involving the direct participation of AREU operators, who tested both the relevance of the responses and the usability of the interface.
The concept behind it is to provide a decision support tool capable of serving a variety of professional roles, from field rescuers to healthcare personnel, and even logistics managers.
In summary, the introduction of this technology at AREU aims to transform the way information is accessed during emergencies, moving from a manual and potentially slow search process to an intelligent dialogue with corporate documentation, thereby supporting faster, better-informed, and safer decision-making for the public.

= State of the Art: Retrieval-Augmented Generation (RAG) Systems

== Genesis of the RAG Paradigm

In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities in learning both factual and semantic knowledge from vast amounts of textual data. However, these models exhibit significant structural limitations, including difficulties with continuous updating, limited interpretability, and a tendency to produce incorrect or unverifiable information—a phenomenon referred to as “hallucination”. These limitations stem from the parametric nature of the knowledge encoded in the models: it is embedded in the weights as the result of a large-scale optimization process and remains largely static.

From a mathematical perspective, this knowledge corresponds to a point (or region) of minimum in the loss function within the parameter space. Consequently, modifying it in a targeted manner requires a new training phase or re-optimization, as it is not possible to adjust individual pieces of content locally without affecting the global balance of the learned parameters.

To address these challenges, the Retrieval-Augmented Generation (RAG) paradigm was introduced by Lewis et al. in 2020 @lewis2020rag. In this framework, the model’s parametric knowledge is complemented by a non-parametric external memory, composed of collections of pre-indexed documents.

== Architecture and Components of a RAG System

A RAG system is based on three fundamental components:
- retrieval from external sources
- knowledge integration
- text generation

Access to external knowledge sources is achieved through semantic retrieval techniques, which go beyond simple lexical matching of keywords by representing both queries and documents as vectors in a continuous semantic space (embeddings). In this vector space, the similarity between texts is computed based on the distance or angle between their respective vectors (e.g., using cosine similarity), enabling the retrieval of relevant documents even when they do not share exact terms with the query but convey the same conceptual content. This approach allows the model to dynamically incorporate updated or specialized information without modifying its internal parameters.

The component typically employed for this phase in a RAG system is the neural retriever, which consists of one or more neural encoders (often based on Transformer architectures) trained to map queries and documents into dense vectors within a shared semantic space. In the most common bi-encoder configuration, one encoder processes the query while another processes the documents (sometimes sharing weights). Each text is thus mapped to a vector $f(x) in RR^d$, where the dimension $𝑑$ is predetermined.

The Transformer architecture, introduced by Vaswani et al. (2017) @vaswani2017attention, relies entirely on attention mechanisms, fully eliminating recurrences and convolutions. Its core component, self-attention, enables each element of a sequence to dynamically weigh all other elements, modeling long-range dependencies in parallel. Given a set of queries $Q$, keys $K$, and values $V$, the scaled dot-product attention is defined as:

$
  attention(Q, K, V) = softmax((Q K^TT) / sqrt(d_k)) V
$

where $d_k$ is the size of the keys. The normalization factor
$1/sqrt(d_k)$ stabilizes the gradient for large vector dimensions.

Instead of applying a single attention function with keys, values, and queries of size $d_upright("model")$, it is advantageous to project linearly
the queries, keys, and values $h$ times through different learned linear projections, respectively towards spaces of size $d_k$, $d_k$, and $d_v$. Multi-head attention enables the model to simultaneously attend to information from different representation subspaces at various positions. In contrast, with a single attention head, the averaging operation tends to limit this capability:

$
  mh(Q, K, V) = concat(head_1, ..., head_h) W^O
$

$
  upright("where") quad head_i = attention(Q W_i^Q, K W_i^K, V W_i^V)
$

and $W_i^Q, W_i^K, W_i^V$ and $W^O$ are learned parametric matrices.

Thanks to this architecture, the Transformer enables fully parallel processing of sequences and efficient modeling of global dependencies, forming the foundation of modern large language models.

The loss function serves as the mathematical criterion guiding the training of the retriever. It is typically contrastive: given a set of relevant (positive) and non-relevant (negative) query-document pairs, the loss is designed to maximize the similarity between vectors of positive pairs while minimizing that of negative pairs. Formally, the encoder parameters are optimized so that the geometric structure of the vector space reflects a notion of semantic relevance. In this way, the learned function induces a metric in the embedding space that approximates the concept of relevance relative to the query.

The most relevant documents identified by the retriever are then provided as context to the generative model, which leverages them to produce the answer. This integration of semantic retrieval and text generation allows the model’s parametric knowledge to be enriched with updatable external information, enhancing performance in knowledge-intensive tasks.

This paradigm has been shown to improve the factual accuracy, transparency, and updatability of LLM-based systems, while reducing the need for continuous retraining. Moreover, the ability to trace information sources during generation represents a key factor for trust and verifiability of responses @rag_survey2023.

== Evolution of RAG Paradigms

Recent literature highlights a progressive evolution of RAG architectures, marked by increasing modularity and a closer integration between retrieval and generation. Early implementations followed a linear pipeline, in which a query was transformed into an embedding, used to retrieve documents, and then fed to the generative model @rag_survey2023.

In this initial phase, retrieval mechanisms were largely derived from traditional Information Retrieval: lexical approaches such as BM25 @okapi_trec1994 were gradually complemented and eventually replaced by dense neural models based on vector representations shared between queries and documents. The adoption of semantic embeddings enhanced both the robustness and generalization capabilities of these systems.

As the paradigm matured, research introduced modular improvements aimed at optimizing the quality of context provided to the generator. These include strategies for document chunking (segmenting into coherent informative units), query expansion, dynamic context selection, re-ranking, and semantic filtering. Such techniques reflect the recognition that the notion of relevance in generative systems differs from that in traditional retrieval systems @survey2025.

A further evolution concerns the integration of retrieval and reasoning.
Modern architectures incorporate multi-hop approaches, in which the system performs multiple iterations of retrieval and reasoning, as well as query decomposition strategies. This enables the handling of complex tasks that require distributed inferences across multiple documents or heterogeneous sources.

From this perspective, a key aspect is the alignment between the retriever and the generator. Zhou & Chen (2025) propose an end-to-end optimization of the retriever based on in-context relevance, demonstrating that relevance for generative contexts differs from that in traditional retrieval systems @openrag2025. Their approach highlights how the efficiency of retrieval can significantly impact overall performance, even when using smaller generative models.

Simultaneously, recent studies underscore the importance of managing the context window of LLMs. Techniques such as semantic compression, adaptive document selection, and noise control aim to optimize the use of available context. In particular, Cuconasu et al. (2024) show that appropriate filtering and weighting of retrieved documents reduces noise and improves model effectiveness @noise2024, while Cheng et al. (2025) provide a systematic overview of the most recent strategies for efficiently combining and selecting documents @survey2025.

A counterintuitive finding in the literature concerns the role of noise in the retrieval process. Cuconasu et al. (2024) demonstrate that the controlled introduction of partially irrelevant documents can actually improve the accuracy of the generative model, suggesting that a more diverse context facilitates the inferential process @noise2024. A possible theoretical explanation is that moderately heterogeneous information acts as a form of contextual regularization: including documents that do not strictly overlap with the query may reduce the risk of overfitting the model to partial or redundant evidence, promoting a more robust latent representation. Moreover, a diversified context can expand the semantic space explored by the model during generation, increasing the likelihood of activating relevant but not immediately obvious conceptual connections.

Another hypothesis is that controlled noise improves attention calibration: the need to discriminate between strongly and weakly relevant signals may lead the model to weigh the available information more selectively, reinforcing the integration and comparison mechanisms across sources. In this view, the observed positive effect does not arise from the noise itself but from its role in stimulating selection and inferential composition within the context window. This finding challenges traditional retrieval evaluation metrics and opens new perspectives for the design of RAG systems.

Finally, there is growing interest in integrating multimodal systems, structured databases, and knowledge graphs, extending the role of retrieval beyond simple access to textual documents and positioning RAG systems as hybrid infrastructures for knowledge access and composition.

== Agentic Frontiers, Evaluations, and Future Challenges

Another research direction concerns computational efficiency and scalability. The adoption of distributed architectures, incremental indexing, and semantic caching enables the deployment of RAG systems in real-world contexts characterized by latency and cost constraints. Furthermore, integration with agentic systems and automated workflows represents a rapidly growing area. By agentic system, we refer to an architecture in which the language model is not limited to generating a response from a single prompt, but functions as an agent capable of planning, making intermediate decisions, and interacting with external tools (e.g., search engines, databases, APIs, or computation modules).

The cutting edge of research is shifting toward Multi-Agent Systems (MAS). In this paradigm, task complexity is not handled by a single “generalist agent,” but distributed across a network of specialized agents that collaborate. This decomposition allows assigning distinct roles to different logical entities, such as:
- Research Agent: specializes in semantic query expansion (hypothetical queries) and precise retrieval of chunks.
- Critical or Validator Agent: evaluates the relevance of sources and the coherence of responses, triggering correction cycles if quality criteria are not met.
- Synthesizer Agent: focuses on final text generation, ensuring that outputs are anchored to sources and free from hallucinations.

This shift toward a multi-agent approach provides a theoretical justification for the use of computational graph architectures, where each node represents not a simple function but an agent with its own domain of expertise. The MAS approach enhances both modularity and interpretability: since each agent must provide reasoning for its actions, it becomes possible to trace precisely where errors may occur in the reasoning chain.

Despite progress in defining and implementing efficient and comprehensive RAG systems, several challenges remain, including:
- alignment between retrieval and generation
- efficient knowledge selection
- interpretability of systems
- realistic performance evaluation
- management of obsolete or contradictory information
- definition of realistic benchmarks and appropriate evaluation metrics

== Conclusion

The RAG paradigm has evolved from a simple document retrieval system into a complex, modular, and knowledge-oriented ecosystem. The dynamic integration of external knowledge enhances the reliability, factual accuracy, and transparency of language models, making these systems increasingly suitable for real-world applications in specialized domains.

Future directions include multimodal systems, autonomous agents, adaptive retrieval, integration with knowledge graphs, and advanced reasoning strategies. These innovations have the potential to enable the development of more robust, interpretable, and updatable intelligent systems.

= Methods

== System Architecture

=== Selection of Database and Models

For the implementation of the system, a targeted selection was made regarding both the database used for storing information and the models employed in the retrieval and generation phases.
In particular, PostgreSQL was chosen as the database management system, enhanced with pgvector, an extension that enables efficient storage and search of vector embeddings. This solution allows the integration of vector search capabilities directly within a relational database, simplifying the system’s infrastructure and facilitating the management of documents and their embeddings used in the retrieval process.

Regarding artificial intelligence models, the Amazon Bedrock service was employed, providing access to a variety of foundational models through a unified interface. Using this platform enables the integration of models for both embedding and generation, maintaining a flexible and scalable architecture while facilitating experimentation and the replacement of models without substantial modifications to the system infrastructure.
Furthermore, leveraging Amazon Bedrock allows for controlled access to models via the Amazon Web Services infrastructure, ensuring greater data governance and helping to preserve the confidentiality of documents used by the system — a particularly important consideration in the context of this work.

=== LangGraph

For the implementation and orchestration of the RAG system workflow, LangGraph was selected — a library designed for building applications based on Large Language Models (LLMs) using graph-based computational structures.
This choice stems from the need to define a structured processing pipeline that enables dynamic integrations across different phases, while providing modularity and flexibility in execution, thereby ensuring the robustness required in a critical context such as AREU.
LangGraph extends the paradigm of linear pipelines commonly used in LLM frameworks, allowing for the definition of more complex execution flows characterized by conditional transitions and iterative cycles.

Unlike traditional sequential architectures, in which operations follow a static pipeline, LangGraph represents the process as a directed graph composed of nodes and edges.
In this framework, each node represents a computational unit — such as a retrieval, reasoning, or generation module — while the edges define the flow of data and the conditions for transitioning between the different system phases. In practice, each node functions as an autonomous agent executing a specific role.

Another advantage offered by LangGraph is the ability to assign different models to each node based on the type of task to be performed. This means that, depending on the complexity and criticality of the operation, it is possible to use more specialized and high-performance models or lighter, less expensive versions. For example, simple data processing or filtering tasks can be handled by more economical models, while complex response generation or advanced reasoning can leverage more powerful models. This flexibility allows for a balanced optimization between computational cost and performance quality, dynamically allocating resources according to the specific needs of each node. Moreover, it facilitates the integration of new models or technological updates without the need to redesign the entire workflow, preserving the modularity and scalability of the architecture.

Processing is further guided by a shared graph state, a data structure containing relevant information accumulated during execution. This state is passed between agents and progressively updated, allowing the system to track the current context of a request and the intermediate results generated by the various workflow components.

The adoption of a graph-based architecture has enabled the implementation of more sophisticated reasoning strategies, including iterative cycles between retrieval and generation, quality checks on produced responses, and fallback mechanisms when retrieved information is insufficient.
Simultaneously, this structure ensures high modularity: each graph node encapsulates a specific system functionality, allowing individual components — such as the retriever or generative model — to be modified or replaced without affecting the overall workflow.
These features enhance system flexibility, supporting both dynamic adaptation during execution and experimentation with alternative architectural configurations.

=== Pipeline Description

The workflow of the RAG architecture is illustrated in @fig:workflow, which depicts all phases of the process.
Upon receiving a query, an agent determines whether to terminate the conversation, generate a direct response without retrieving documents, or proceed with retrieval. This strategy avoids unnecessary transitions to the retrieval phase for questions that do not require document consultation, thereby optimizing both computational costs and response times.

In the retrieval scenario, another agent generates two questions semantically related to the original query (hypothetical queries), embeds all three queries, and computes the average of the three embeddings to retrieve the most relevant documents from the database. The retrieved documents are then provided as context to the generative agent, which produces the final response.

The subsequent step involves validating the generated response, comparing it with the original query to assess the coherence and correctness of the provided information. If the outcome is negative, the system may iterate the retrieval process, updating the graph state with the newly obtained information and generating additional queries to refine the search, or terminate the conversation if further attempts are unlikely to significantly improve the response (e.g., due to insufficient documentation).

When the response is deemed satisfactory, the system returns a contextualized answer to the end user, grounded in external sources, thereby enhancing the reliability and relevance of the information compared to a standalone generative model.

All elements generated during the process — queries, retrieved documents, produced responses, and the time taken by each node — are stored in the graph state, allowing the system to maintain a complete trace of the interaction. This information can be used in subsequent iterations or for post-hoc analyses of system performance.
Furthermore, every step is accompanied by reasoning, a textual explanation describing the decisions made by the system at each phase, thereby improving transparency and interpretability of the process.

#figure(
  image("/ObservableRagAgentTest-graph.png", width: 100%),
  caption: [RAG workflow],
) <fig:workflow>

== Dataset and Preprocessing

The documents used for retrieval were collected from both public and private sources and include operational instructions, internal procedures, and other relevant materials for emergency management.
The preprocessing pipeline involved data cleaning, text normalization, and segmentation into coherent informative units (chunking). Each sufficiently long chunk is accompanied by a summary and three hypothetical queries — automatically generated questions whose answers are contained within the chunk and represent different informational aspects — to improve semantic coverage during retrieval. It has been observed that including hypothetical queries alongside the textual content of the chunk significantly enhances retrieval performance.

All chunks have been further enriched with keywords derived from the general context of the document, ensuring that relevant information, even when distributed across different sections of the text, is preserved. This approach guarantees that crucial contextual knowledge is not lost, facilitating accurate and comprehensive retrieval.

The database was implemented using PostgreSQL with the pgvector extension, enabling efficient storage and indexing of vector embeddings associated with the documents.
The embeddings were generated using Titan Text Embeddings V2, an embedding model available on Amazon Bedrock, and stored in the database together with document metadata, such as titles and the distinction between public and private documents.

== Retrieval with Reasoning

During the retrieval phase, the system generates two hypothetical queries based on the user’s question, transforms them into embeddings, and computes the average of the three embeddings to retrieve the most relevant documents from the database. By considering documents closest to the semantic mean of the three related queries, the system more accurately identifies the chunks that are truly useful for generating the response. Subsequently, the content of the retrieved documents is evaluated to determine the relevance and pertinence of the information with respect to the original query. This evaluation is performed by a model that assigns each chunk a score between 0 and 1, representing the probability of relevance, along with a textual explanation justifying the assessment.

After this step, documents are ranked by relevance and filtered to retain only the most pertinent ones using a sigmoid function centered at the average score with a steepness of 8. This approach reduces noise and improves the quality of the context provided to the generative model. The raw relevance scores are first transformed via the sigmoid function before filtering. Consequently, only documents with a sigmoid value greater than 0.65 are retained, while those with a value ≤ 0.65 are discarded, as illustrated in @fig:sigmoid. This re-ranking and filtering strategy allows non-relevant chunks to be discarded without risking the loss of useful information.

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

Finally, the selected documents are provided as context to the generative model, retaining information about their relevance to the original query. This allows the agent to integrate the information more effectively during the response generation process.

== Generation and Validation

The generative model leverages the retrieved documents along with their relevance scores, the original query, and the corresponding hypothetical queries to produce a contextualized, source-based response. Response generation is guided by a Jinja2 prompt, designed to encourage the model to integrate information coherently and provide explanations for the decisions made during generation. The use of Jinja2 also enables dynamic prompt structuring, facilitating the insertion and management of variables—such as documents and queries—in a modular, reusable, and maintainable manner. The response includes citations to the documents used, a summary of the content, and an explanation of the inferential process followed by the model to arrive at the final answer.

The validation phase directs the system’s flow, determining whether to iterate retrieval or terminate the conversation. In this step, the agent provides a textual explanation describing the rationale behind its decision, enhancing the transparency of the process and allowing the user to understand why the response was deemed satisfactory or not. Specifically, the model assigns a satisfaction score between 0 and 1, calculated according to predefined criteria; a response is considered acceptable when the score is ≥ 0.8. If the score is < 0.8, the system evaluates whether to iterate the retrieval process, updating the graph state with the newly obtained information and generating additional queries to refine the search, or to terminate the conversation if further attempts are unlikely to significantly improve the response (e.g., due to insufficient documentation).

= Results

Preliminary tests were conducted to evaluate the effectiveness of the implemented RAG system, focusing on semantic retrieval quality, response generation quality, and search times. The results indicate that the system is capable of retrieving relevant documents even in the presence of complex or ambiguous queries, thanks to the use of semantic embeddings and the generation of hypothetical queries. The responses produced are contextualized and grounded in reliable sources, with a significant reduction in search times compared to a purely manual approach.

The most impactful factor in retrieval quality was the inclusion of hypothetical queries: in most cases, adding the appropriate question was sufficient to correctly retrieve the most relevant chunk. Additionally, a well-structured user input further improves retrieval performance. Regarding response generation, particular attention was given to prompt engineering, as it emerged that the prompt is a critical determinant of output quality. Given the same informative context and a well-defined query, the structure, clarity, and instructions contained in the prompt significantly influence the coherence, completeness, and reliability of the generated response. Moreover, a clear and precise question enables the model to better identify which information to include, reducing ambiguity and improving the relevance of the output.

AREU operators were directly involved in testing, providing qualitative feedback on the relevance of responses and the usability of the system. To automate the evaluation process and establish a benchmark for future developments, a dataset of questions accompanied by the expected content for their respective answers was constructed. Preliminary results suggest that RAG systems hold significant potential in emergency contexts, where the speed and reliability of information are critical.

To evaluate the impact of model quality and optimize resource usage, only the model in the node responsible for response generation was modified. In the other nodes, the same model was retained, as less complex tasks do not require the use of more powerful and costly models. The three generative models compared were Amazon Nova 2 Lite, Claude Sonnet 4.6, and Mistral Pixtral Large. As shown in @fig:scores_comparison, Sonnet achieved the best overall performance, standing out with the highest number of perfect responses, while Pixtral demonstrated the weakest performance. However, the differences between Sonnet and Nova were less pronounced. This suggests that retrieval quality and prompt structure can have a more decisive impact on overall performance than the specific choice of generative model, at least within the range of models tested. These results highlight the importance of optimizing retrieval components and prompt engineering to maximize the effectiveness of RAG systems, rather than focusing exclusively on the selection of the generative model.

#figure(
  image("/hist_side_by_side.png", width: 100%),
  caption: [Scores comparison],
) <fig:scores_comparison>

Regarding response times, @fig:boxplot_times shows a significant difference in latency between the three models, with Amazon Nova 2 Lite returning responses considerably faster than Claude Sonnet 4.6 and Mistral Pixtral Large. This result underscores the importance of considering not only the quality of generated responses but also the computational efficiency of the models, especially in contexts such as AREU, where timely access to information is critical.

#figure(
  image("/boxplot_times.png", width: 100%),
  caption: [Boxplot times],
) <fig:boxplot_times>

A more detailed analysis of execution times across the other nodes of the graph (@fig:nodes_times_comparison) reveals that the re-ranking phase constitutes the primary bottleneck, with execution times significantly higher than those observed in the other phases. This finding emphasizes the need to optimize the evaluation and selection of retrieved documents, as it represents a critical factor for the overall efficiency of the system. Improvements in this phase could lead to substantial reductions in response times, making the system more suitable for real-world scenarios where rapid access to information is essential.

#figure(
  image("/nodes_times_comparison.png", width: 100%),
  caption: [nodes times comparison],
) <fig:nodes_times_comparison>

= Discussion

The implemented RAG system has proven effective in retrieving relevant information and generating contextualized responses, thereby improving the reliability and relevance of the information provided to AREU operators. However, several limitations have emerged, particularly the need for further optimization to handle more complex and dynamic scenarios. Notably, the documents used for retrieval were not originally designed for processing within a RAG framework, and therefore exhibit structures that are not optimal for both retrieval and generation tasks.

In particular, the presence of redundant information (i.e., repeated concepts across sections or documents), obsolete content (outdated directives), and unstructured elements (such as images and tables) complicates the system’s ability to identify and integrate the most relevant information. This highlights the importance of a more targeted data curation and organization process to maximize the effectiveness of RAG systems. Designing documents specifically for RAG usage would reduce the number of chunks passed to the re-ranking phase, thereby improving overall performance, as the system would require less time both to process and to reason over the content.

Among the strengths of the system, the modularity of the architecture—a key feature of the multi-agent approach—stands out, as it allows individual components to be updated or replaced without affecting the overall workflow. Additionally, the ability to dynamically integrate external knowledge enhances the factuality and transparency of generated responses. The adoption of a graph-based architecture has also enabled the implementation of more advanced reasoning strategies and the maintenance of a complete trace of interactions, facilitating both iterative refinement and post-hoc analysis of system performance.

Future developments include the integration with existing healthcare information systems, with the aim of further improving the relevance and practical utility of responses for AREU operators. Another promising direction involves the exploration of multimodal approaches, incorporating non-textual sources such as images and structured data to enrich the context available to the generative model. Finally, further research will focus on the systematic comparison of different embedding and generation models, as well as the exploration of hierarchical ranking strategies to optimize document selection during retrieval.

In conclusion, RAG systems demonstrate significant value in emergency contexts, where both the speed and reliability of information are critical. By grounding responses in verified documentation, such systems enhance operational efficiency and support more informed decision-making. Their flexibility and adaptability make them suitable for a wide range of professional roles—from field rescuers to healthcare personnel and logistics managers—ultimately contributing to faster, safer, and more effective interventions.

#bibliography("bibliography.bib", style: "ieee")
