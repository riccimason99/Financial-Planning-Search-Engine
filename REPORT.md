# Information Retrieval System Report
**INFO 624 - Information Retrieval Final Project**  
**Financial Planning Search Engine**  
**Mason Ricci**

---

## What, Why and Who

This financial planning search engine is designed to help users find relevant, actionable information from a diverse corpus of financial literacy materials. This is designed to be an educational tool for people trying to manage their own finance. The a RAG system provides the security of using a known knowledge base with the convience of a natural language response using an OpenAi api call. 

The system indexes content from multiple authoritative sources including the Consumer Financial Protection Bureau (CFPB), the Council for Economic Education's National Standards for Personal Financial Education, the Securities and Exchange Commission (SEC), university textbooks on personal finance, and plain-language financial planning guides. While these materials represent high-quality educational resources in the financial domain, the primary purpose of this project is to demonstrate the effectiveness of the semantic search engine itself, rather than to showcase the corpus quality. Indeed, one of the most powerful features of this retrieval system is its domain flexibility—the same architecture and indexing approach can be applied to any corpus, whether medical literature, legal documents, technical manuals, or any other text-based knowledge domain.

The system addresses three distinct use cases that represent common information needs in personal finance. These use cases were selected to test the search engine's ability to retrieve relevant information across different financial planning topics: investing, budgeting, and debt management. Each use case represents a realistic scenario where a user needs targeted guidance from the corpus.

#### Use Case 1: Investment Education

**Information Need:** Users seeking to begin investing need foundational knowledge about different investment types, particularly the differences between stocks and bonds, basic asset allocation principles, and practical steps to start building an investment portfolio. The information need spans both conceptual understanding (what are stocks and bonds, how do they work) and practical guidance (how to begin investing, what to consider when choosing investments).

**Query:** "How do I start investing in stocks and bonds?"

**Search Challenge:** This query requires the system to retrieve educational content that explains investment basics, compares different asset classes, and provides actionable guidance. The retrieved chunks should cover both conceptual understanding and practical steps, balancing introductory material with enough depth to enable informed decision-making.

### Use Case 2: Budget Creation and Maintenance

**Information Need:** Users attempting to manage their finances need concrete budgeting strategies including expense tracking methods, tools for monitoring spending, techniques for staying within budget limits, and approaches to handle both regular and irregular income. The information need encompasses the mechanics of building a budget (what categories, how to track) and behavioral strategies to maintain budget discipline over time.

**Query:** "What is the best way to create and stick to a monthly budget?"

**Search Challenge:** This query requires the system to retrieve both tactical information (how to build a budget, what tools to use) and behavioral guidance (how to maintain discipline). The ideal chunks should provide practical frameworks and step-by-step processes rather than just theoretical budgeting concepts.

### Use Case 3: Credit Card Debt Resolution

**Information Need:** Users struggling with high-interest credit card debt need specific debt payoff strategies, particularly methods designed for managing multiple debts simultaneously. The information need includes understanding different debt reduction methodologies (avalanche vs. snowball methods), tactics for reducing interest rates through negotiation or consolidation, strategies for finding extra money to accelerate payments, and realistic plans for systematic debt elimination while maintaining necessary living expenses.

**Query:** "What strategies can I use to pay off my credit card debt?"

**Search Challenge:** This query requires retrieval of actionable debt reduction strategies with specific focus on credit cards and high-interest debt. The system should surface chunks that explain different methodologies, compare their effectiveness, and provide tactical advice for debt prioritization and payment allocation. Relevant chunks may also include negotiation tactics and behavioral strategies for avoiding additional debt accumulation.

---

These three use cases represent distinct information retrieval challenges: educational content (investing), procedural guidance (budgeting), and problem-solving strategies (debt elimination). By evaluating system performance across these diverse query types, we can assess the search engine's ability to retrieve relevant information for different user intents within the financial planning domain.

---

## How 

This section describes the technical implementation of the information retrieval system, covering document processing, indexing, query processing, and retrieval mechanisms.

### Document Processing and Indexing

The indexing pipeline transforms raw documents into searchable vector representations through several stages. First, documents are extracted from PDF files using PyMuPDF (fitz), which converts each page into plain text while preserving document structure. These extracted texts are then processed by LangChain's RecursiveCharacterTextSplitter, which segments the documents into overlapping chunks. The chunking parameters were set to 1000 characters per chunk with 200 characters of overlap between consecutive chunks. This overlap ensures that important information spanning chunk boundaries is not lost, while the 1000-character size provides sufficient context for semantic understanding without overwhelming the embedding model or retrieval results.

The choice of chunk size represents a key trade-off in the system design. Smaller chunks (e.g., 500 characters) would provide more precise retrieval but might lack sufficient context for the embedding model to capture semantic meaning. Larger chunks (e.g., 2000 characters) would provide more context but could conflate multiple topics within a single chunk, reducing retrieval precision. The 1000-character size with 200-character overlap strikes a balance between semantic coherence and retrieval granularity, typically capturing 2-3 paragraphs of content per chunk.

Each text chunk is then converted into a dense vector representation using OpenAI's text-embedding-3-small model, which generates 1536-dimensional embeddings. This embedding model uses a transformer-based architecture trained on diverse text corpora to capture semantic relationships between words and concepts. Unlike traditional keyword-based methods that rely on term frequency statistics, these dense embeddings encode meaning in a continuous vector space where semantically similar concepts are positioned closer together. The embedding process occurs once during indexing, and the resulting vectors are stored in a FAISS (Facebook AI Similarity Search) index on local disk at `faiss_index/`.

The current corpus, consisting of nine financial planning documents, is segmented into 4,329 distinct chunks, each represented as a 1536-dimensional vector in the FAISS index. FAISS provides efficient similarity search capabilities using optimized algorithms for nearest neighbor retrieval in high-dimensional spaces. For this project, we use exact search (exhaustive k-NN) rather than approximate methods, ensuring that the top-k results are guaranteed to be the most similar vectors in the index.

### Query Processing and Retrieval

When a user submits a query, it undergoes the same embedding process as the indexed documents. The query text is passed to the text-embedding-3-small model, generating a 1536-dimensional query vector. This ensures representation consistency between queries and documents, a critical requirement for effective similarity-based retrieval.

The query vector is then compared against all document vectors in the FAISS index using cosine similarity as the distance metric. Cosine similarity measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical), with higher values indicating greater semantic similarity. FAISS implements this efficiently by using L2 distance on pre-normalized vectors, which is mathematically equivalent to cosine similarity but computationally faster. The system retrieves the top k document chunks (k=5 by default) with the highest similarity scores, ranked in descending order of relevance.

The ranking approach is purely based on vector similarity without additional re-ranking or query expansion. This represents a straightforward neural retrieval model where relevance is determined entirely by the semantic proximity between query and document embeddings. While this approach may miss some relevant documents that use different terminology, the semantic nature of the embeddings generally captures synonym relationships and conceptual similarity that keyword-based methods would miss.

### Retrieval-Augmented Generation (RAG)

After retrieving the top-k most relevant chunks, the system employs Retrieval-Augmented Generation to provide natural language responses. The retrieved document chunks are concatenated and provided as context to OpenAI's GPT-4o-mini model along with the user's original question. This grounds the language model's response in the actual corpus content, reducing hallucination and ensuring answers are based on the indexed documents rather than the model's pre-training knowledge.

The system supports two interaction modes: single-turn queries (via `ask.py`) where each question is independent, and multi-turn conversations (via `chat.py`) where conversation history is maintained. In multi-turn mode, the ConversationManager preserves the last 20 message pairs, enabling the system to resolve references (e.g., "What are the limits for it?" following a question about 401(k) plans) and maintain contextual coherence across the conversation.

### Technical Specifications

**Embedding Model:** OpenAI text-embedding-3-small (1536 dimensions)  
**Vector Store:** FAISS with exact k-NN search  
**Chunk Size:** 1000 characters with 200-character overlap  
**Similarity Metric:** Cosine similarity (via L2 distance on normalized vectors)  
**Retrieval:** Top-5 chunks by default, configurable up to k=10  
**Generation Model:** GPT-4o-mini with temperature 0.7  
**Corpus Size:** 9 documents, 4,329 indexed chunks  
**Index Storage:** Local disk-based FAISS index (~5MB)  

The system is implemented entirely in Python using OpenAI's API for embeddings and generation, LangChain for document processing and FAISS integration, and PyMuPDF for PDF extraction. All components run locally without requiring external databases or services beyond the OpenAI API endpoints.

---

## Evaluation Methodology

To assess the effectiveness of the retrieval system, we employed standard Information Retrieval evaluation metrics using manually-created relevance judgments. The evaluation process consisted of three stages: query execution, relevance assessment, and metric calculation.

### Query Execution and Result Set

For each of the three test queries, the system retrieved the top 5 most similar document chunks from the corpus using the semantic search pipeline described above. The decision to retrieve k=5 chunks. Since retrieved chunks serve as context for a language model to generate natural language responses, limiting retrieval to 5 chunks serves two purposes: first, it reduces information overload for both the end user and the generation model, focusing on the most relevant information rather than diluting quality with marginally relevant content; second, it maintains reasonable API costs and response latency, as each additional chunk increases both the embedding computation and the context length sent to the generation model. For educational and informational queries, precision is more valuable than exhaustive recall—users benefit more from highly relevant, focused information than from comprehensive but scattered results. We evaluete the chunks that are returned before they are sent to the LLM for a final natural language response. 

### Relevance Assessment

After retrieving the top 5 chunks for each query, relevance judgments were created through manual review of the retrieved content. Each chunk was assessed for its ability to answer the query and assigned a relevance score on a three-point scale: 3 (highly relevant) for chunks that directly address the query and provide core information needed to answer it; 2 (somewhat relevant) for chunks that contain related information but are not central to answering the query; 1 (marginally relevant) for chunks with tangential connections to the query; and 0 (not relevant) for chunks that do not contribute to answering the query. Chunks scored as 0 were excluded from the relevant set entirely.

The relevance assessment considered whether each chunk contained information that would help a user answer their question, regardless of completeness. For example, a chunk explaining the snowball method of debt repayment was judged highly relevant (3) for Query 3 even if it didn't mention credit cards specifically, because the strategy applies directly to credit card debt payoff. Conversely, chunks consisting only of table-of-contents entries or page headers were marked as not relevant (0) despite containing keywords from the query, as they provided no substantive information.

While this approach requires subjective judgment, it represents the only feasible method for determining whether retrieved information truly satisfies an information need—no automated metric can assess semantic relevance without human-labeled ground truth.

**Important Limitation:** For practical evaluation with a corpus of 4,329 chunks, we could only assess the relevance of chunks that were actually retrieved by the system. Computing true recall would require manually reviewing all 4,329 chunks for each query to identify every relevant chunk in the corpus—a prohibitively time-consuming task. Therefore, our evaluation makes relies on Precision@5 and nDCG@5 as metrics to determine the quality of results.

### Evaluation Metrics

**Precision@5** measures the proportion of retrieved chunks that are relevant. It answers the question: "Of the 5 chunks returned, how many are actually useful?" Precision is calculated as the number of relevant chunks in the top 5 divided by 5. This metric is fully reliable as it depends only on assessing the retrieved results, not the entire corpus.

**nDCG@5** Is a graded relevance metric that considers both whether relevant chunks were retrieved and their ranking positions. Unlike binary metrics, nDCG accounts for the degree of relevance (scores 1-3) and applies a logarithmic discount to lower-ranked positions, reflecting the reality that users pay more attention to top results. nDCG ranges from 0 to 1, with 1 indicating perfect ranking where all highly relevant chunks appear at the top positions.

---

## Evaluation Results

The retrieval system demonstrated strong performance across all three test queries, achieving high precision and excellent ranking quality.

### Summary Table

| Query | Precision@5 | nDCG@5 |
|-------|-------------|--------|
| Query 1: Investing | 0.800 | 1.000 |
| Query 2: Budgeting | 1.000 | 0.944 |
| Query 3: Debt Payoff | 1.000 | 0.944 |
| **Average** | **0.933** | **0.963** |

### Per-Query Results

**Query 1: "How do I start investing in stocks and bonds?"**
- Precision@5: 0.800 (4 of 5 retrieved chunks were relevant)
- nDCG@5: 1.000 (perfect ranking with highly relevant chunks at top positions)
- One chunk contained information from a table of contents which was not useful

**Query 2: "What is the best way to create and stick to a monthly budget?"**
- Precision@5: 1.000 (all 5 retrieved chunks were relevant)
- nDCG@5: 0.944 (near-perfect ranking quality)

**Query 3: "What strategies can I use to pay off my credit card debt?"**
- Precision@5: 1.000 (all 5 retrieved chunks were relevant)
- nDCG@5: 0.944 (near-perfect ranking quality)

### Aggregate Performance

| Metric | Average Score |
|--------|---------------|
| **Precision@5** | **0.933** (93.3%) |
| **nDCG@5** | **0.963** (96.3%) |

These results indicate that the semantic search system effectively retrieves relevant information across diverse query types. The high precision (93.3%) demonstrates that nearly all retrieved chunks contribute useful information, with minimal noise in the results—specifically, 14 of 15 total retrieved chunks were judged relevant. The high nDCG score (96.3%) confirms that relevant chunks consistently appear at the top of the ranking, with the most relevant information positioned where users are most likely to see it.

The precision results are particularly meaningful in the context of the k=5 retrieval limit. With only 5 opportunities to retrieve useful content per query, achieving 93.3% precision means the system rarely wastes a retrieval slot on non-relevant content. The single non-relevant chunk (a table-of-contents entry in Query 1, position 4) represents a known limitation of text-based retrieval systems that occasionally surface structural elements, but this did not significantly impact overall performance given the highly relevant content in positions 0-2.

---

## Discussion

### Strengths of the Semantic Search Approach

The primary strength of this system lies in its semantic understanding capabilities. Unlike traditional keyword-based search engines that rely on exact term matching (e.g., BM25, TF-IDF), the dense embedding approach captures conceptual similarity independent of specific vocabulary choices. This is particularly valuable in the financial planning domain where users may ask questions using colloquial language ("pay off my credit card debt") while relevant documents use formal terminology ("debt reduction strategies," "unsecured debt elimination"). The evaluation results demonstrate this strength—Query 3 successfully retrieved chunks about "high-interest debt" and "unsecured debt" strategies despite not using the exact phrase "credit card."

The use of FAISS for vector similarity search provides computational efficiency that scales well with corpus size. While the current index contains 4,329 chunks, FAISS can handle millions of vectors with minimal performance degradation when using approximate nearest neighbor methods. The semantic embeddings also eliminate the need for complex query processing steps like stemming, stop-word removal, or synonym expansion, as these linguistic variations are implicitly handled by the embedding model's semantic representations.

The Retrieval-Augmented Generation component adds significant value by transforming ranked chunks into coherent natural language responses. Rather than presenting users with raw document excerpts requiring manual synthesis, the RAG approach provides direct answers while maintaining grounding in the corpus. The conversation memory feature further enhances usability by enabling multi-turn dialogues where users can ask follow-up questions or request clarification without restating context.

### Limitations and Challenges

Despite strong evaluation results, the system exhibits several limitations. First, the retrieval approach relies entirely on semantic similarity without incorporating term-matching signals. This can occasionally cause the system to miss documents that match query terms exactly but are positioned slightly farther away in embedding space due to surrounding context. A hybrid approach combining dense semantic retrieval with sparse lexical matching (BM25) might capture both semantic similarity and exact term matches, potentially improving precision further.

Second, the system has no query understanding or expansion capabilities. Complex queries with multiple information needs (e.g., "What are stocks and bonds and how do I get out of debt?") are embedded as single vectors, which may not optimally match chunks addressing different the sub-questions. 

Finally, the system's dependence on OpenAI's embedding and generation models. The embedding model is also a black box—we cannot inspect or modify how it represents concepts, limiting our ability to debug retrieval failures or bias the system toward domain-specific terminology.


## How to use

This project can be run locally, to do so please reference the USAGE.md file in the repository. 
