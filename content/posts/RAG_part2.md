---
title: 'From Retrieval to RAG (Part - 2) [Draft]'
date: 2024-10-19
draft: false
comments: false
---


The concept of "Retriever + Generator End-to-end Training" (referred to as "RAG") by Lewis et al. (2020) integrates retrieval and generation into a single, cohesive framework. This method enhances the accuracy of generating relevant and accurate responses by training both components together, ensuring that the retriever provides relevant documents and the generator produces high-quality responses. Let's break down the details step by step:

### Components of RAG

1. **Retriever:**
    - The retriever is responsible for searching a large corpus to find documents relevant to the input query.
    - It maximizes the overall likelihood by optimizing mixture weights over the retrieved documents.

2. **Generator:**
    - The generator takes the documents retrieved by the retriever and generates the final output, such as an answer to a question or a piece of text based on the given input.
    - It maximizes the generation likelihood given the single retrieved document.

### Training Process

The training process in RAG involves simultaneous optimization of both the retriever and the generator, which are interconnected through the end-to-end backpropagation mechanism.

1. **Query Encoding ($q(x)$):**
    - The input query is encoded into a query embedding vector $q(z)$.
    - This vector is used to search for relevant documents in the document index.

2. **Document Encoding (enc, $d(z)$):**
    - Each document in the corpus is encoded into an embedding vector $d(z)$.

3. **Document Retrieval ($d(z)$):**
    - The retriever component uses the query embedding $q(z)$ to identify a set of top-$k$ relevant documents $z$ from a large corpus.
    - This process involves measuring the similarity between the query embedding and the document embeddings $d(z)$, often using techniques like Maximum Inner Product Search (MIPS).

4. **Generation (p, G):**
    - The generator takes each retrieved document $z$ and the query embedding $q(z)$ to generate the output text $y$.
    - The generation is modeled as a mixture model: for each query, a document is selected, and then the response is generated based on that document.

### End-to-End Backpropagation

The key innovation in RAG is the end-to-end backpropagation through both the retriever and the generator, allowing the model to be trained jointly.


### Likelihood Maximization

- The retriever and generator are trained to maximize the joint likelihood of the retrieved document and the generated response.
- The overall probability $P_{\text{RAG}}(y|x)$ is approximated by summing over the top-$k$ retrieved documents $z$:

$$
P_{\text{RAG}}(y|x) \approx \sum_{z \in \text{top-}k(p_R(x))} p_R(z|x) \prod_{i} p_G(y_i|x, z, y_{1:i-1})
$$

Here, $p_R$ is the probability distribution of the retriever, and $p_G$ is the probability distribution of the generator.

### Components of RAG

1. **Retriever:**
    - The retriever is responsible for searching a large corpus to find documents relevant to the input query.
    - It maximizes the overall likelihood by optimizing mixture weights over the retrieved documents.

2. **Generator:**
    - The generator takes the documents retrieved by the retriever and generates the final output, such as an answer to a question or a piece of text based on the given input.
    - It maximizes the generation likelihood given the single retrieved document.

### Training Process

The training process in RAG involves simultaneous optimization of both the retriever and the generator, which are interconnected through the end-to-end backpropagation mechanism.

1. **Query Encoding ($q(x)$):**
    - The input query is encoded into a query embedding vector $q(x)$.
    - This vector is used to search for relevant documents in the document index.

2. **Document Retrieval ($d(z)$):**
    - The retriever component uses the query embedding $q(x)$ to identify a set of top-$k$ relevant documents $z$ from a large corpus.
    - This process involves measuring the similarity between the query embedding and the document embeddings $d(z)$, often using techniques like Maximum Inner Product Search (MIPS).

3. **Document Encoding (enc, $d(z)$):**
    - Each document in the corpus is encoded into an embedding vector $d(z)$.

4. **Generation ($p_G$):**
    - The generator takes each retrieved document $z$ and the query embedding $q(x)$ to generate the output text $y$.
    - The generation is modeled as a mixture model: for each query, a document is selected, and then the response is generated based on that document.

### End-to-End Backpropagation

The key innovation in RAG is the end-to-end backpropagation through both the retriever and the generator, allowing the model to be trained jointly.

### Detailed Breakdown

1. **Likelihood Maximization:**

    - The retriever and generator are trained to maximize the joint likelihood of the retrieved document and the generated response.
    - The overall probability $P_{\text{RAG}}(y|x)$ is approximated by summing over the top-$k$ retrieved documents $z$:

    $$
    P_{\text{RAG}}(y|x) \approx \sum_{z \in \text{top-}k(p_R(x))} p_R(z|x) \prod_{i} p_G(y_i|x, z, y_{1:i-1})
    $$

    - **RAG-Sequence Model:**
    - Uses the same retrieved document to generate the complete sequence.
    - Treats the retrieved document as a single latent variable that is marginalized to get the seq2seq probability $p(y|x)$ via a top-$k$ approximation.
    - Concretely, the top $k$ documents are retrieved using the retriever, and the generator produces the output sequence probability for each document, which are then marginalized:

        $$
        P_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-}k(p_R(x))} p_R(z|x) \prod_{i} p_G(y_i|x, z, y_{1:i-1})
        $$

    - **RAG-Token Model:**
    - Can draw a different latent document for each target token and marginalize accordingly.
    - Allows the generator to choose content from several documents when producing an answer.
    - The top $k$ documents are retrieved using the retriever, and then the generator produces a distribution for the next output token for each document, before marginalizing and repeating the process with the following output token:

        $$
        P_{\text{RAG-Token}}(y|x) = \sum_{z \in \text{top-}k(p_R(x))} p_R(z|x) \prod_{i} p_G(y_i|x, z, y_{1:i-1})
        $$

2. **Retriever Training:**
    - The retriever is trained to give higher similarities to helpful documents by adjusting the embeddings.
    - The probability of the retriever $p_R(z|x)$ is computed based on the inner product of the document embedding $d(z)$ and the query embedding $q(x)$:

    $$
    p_R(z|x) \propto \exp(d(z)^T q(x))
    $$

    Where $d(z) = \text{enc}_d(z)$ and $q(x) = \text{enc}_q(x)$.

3. **Generator Training:**
    - The generator is trained to maximize the likelihood of generating the correct response given the retrieved document.
    - The generator's probability $p_G(y|x, z, y_{1:i-1})$ depends on both the query and the retrieved document.

### Challenges and Considerations

1. **Stale Search Index:**
    - One issue with the end-to-end training is that the search index can become stale because the document embeddings $d(z)$ might change during training.
    - This necessitates regular re-indexing or adopting techniques to handle dynamic embeddings.

2. **Efficiency:**
    - The training process can be computationally intensive as it involves computing embeddings for a large corpus and performing retrieval during training.
    - Efficient indexing and retrieval methods are crucial to making this feasible.

### Summary

RAG represents a sophisticated approach to combining retrieval and generation, enabling the system to dynamically and jointly optimize both components. By leveraging end-to-end backpropagation, RAG can significantly improve the relevance and quality of generated responses, making it a powerful method for tasks that require integrating large-scale information retrieval with natural language generation.



