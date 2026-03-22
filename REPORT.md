# Information Retrieval System Report
**INFO 624 - Information Retrieval Final Project**  
**Financial Planning Search Engine**  
**Mason Ricci**

---

## What, Why and Who

This financial planning search engine was built to help users find relevant and useful information from a set of financial literacy documents. The idea is to create an educational tool for people trying to better manage their own finances. Instead of just searching by keywords, the system uses a RAG based setup, which gives the reliability of a known knowledge base while still letting the user ask questions in natural language and get a natural language response back through an OpenAI API call.

The corpus includes content from several strong sources such as the Consumer Financial Protection Bureau (CFPB), the Council for Economic Education’s National Standards for Personal Financial Education, the Securities and Exchange Commission (SEC), university level personal finance textbooks, and plain language financial planning guides. These are good quality sources, but the main point of the project is not really to prove that the corpus is amazing. It is more to show that the search engine itself works well. One of the better parts of this system is that the same general setup could be used on almost any text corpus. It could be financial documents, legal materials, medical literature, technical manuals, really whatever same type of text based domain you want.

To test the system, I focused on three use cases that reflect common personal finance questions. I wanted the system to retrieve relevant information across different areas of financial planning, not just one narrow topic. So I picked use cases around investing, budgeting, and debt management. These are all realistic questions that somebody might actually ask.

### Use Case 1: Investment Education

**Information Need:** Users need foundational knowledge about stocks and bonds, including their differences, and practical steps to begin investing.

**Query:** "How do I start investing in stocks and bonds?"

**Search Challenge:** This query requires the system to retrieve chunks that explain investment basics, compare asset types, and also give practical advice. Ideally, the results should not just define stocks and bonds, but also help the user understand what to do next.

### Use Case 2: Budget Creation and Maintenance

**Information Need:** Users need concrete budgeting strategies including expense tracking methods, tools for staying within budget limits, and behavioral techniques to maintain discipline.

**Query:** "What is the best way to create and stick to a monthly budget?"

**Search Challenge:** This query needs both tactical and behavioral information. The system should return chunks that explain how to build a budget, what tools or methods can be used, and how someone can stay consistent over time. Just giving abstract budgeting concepts would not really be enough.

### Use Case 3: Credit Card Debt Resolution

**Information Need:** Users need specific debt payoff strategies for credit cards, including comparison of different methods (avalanche vs. snowball), tactics for reducing interest rates, and approaches for accelerating payments.

**Query:** "What strategies can I use to pay off my credit card debt?"

**Search Challenge:** This query requires the system to retrieve practical debt reduction strategies with a focus on high interest debt. The results should explain different payoff approaches, compare them a little, and give advice that a person could realistically act on.

---

These three use cases represent different retrieval challenges. The investing query is more educational, the budgeting query is more procedural, and the debt query is more problem solving focused. Looking at all three makes it easier to judge whether the system can handle different user intents within the same financial planning domain.

---

## How

This section explains the technical implementation of the retrieval system, including document processing, indexing, query handling, and retrieval.

### Document Processing and Indexing

The indexing pipeline takes raw documents and turns them into searchable vector representations. First, the text is extracted from PDF files using PyMuPDF (`fitz`), which converts each page into plain text while keeping most of the document structure. After that, the extracted text is split into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter`. I set the chunk size to 1000 characters with 200 characters of overlap between chunks. The overlap matters because sometimes important information continues across chunk boundaries, and without overlap that can get lost.

The chunk size is basically a tradeoff. Smaller chunks would probably improve precision a bit, but they might not include enough context for strong semantic matching. Bigger chunks would include more context, but then they may mix too many ideas together and hurt retrieval quality. So 1000 characters with 200 overlap felt like a pretty reasonable middle ground. In most cases, that gives around two or three paragraphs of text per chunk.

Each chunk is then embedded using OpenAI’s `text-embedding-3-small` model, which produces a 1536 dimensional vector. These embeddings are meant to capture semantic meaning, not just keyword overlap. So instead of matching exact terms only, the system can retrieve chunks that are conceptually related even if the wording is different. Once the embeddings are created, they are stored in a FAISS index on local disk at `faiss_index/`.

The current corpus contains nine financial planning documents, which were split into 4,329 total chunks. Each chunk is stored as a 1536 dimensional vector in the FAISS index. FAISS is used because it makes nearest neighbor search in high dimensional vector space fast and practical. For this project, I used exact search rather than approximate search, so the top results returned are the actual nearest vectors in the index.

### Query Processing and Retrieval

When the user enters a query, the query goes through the same embedding process as the documents. The text is passed into the same `text-embedding-3-small` model, which generates a 1536 dimensional query vector. Using the same model for both documents and queries is important, because the vectors need to exist in the same semantic space.

The query embedding is then compared against all chunk embeddings in the FAISS index using cosine similarity. Higher cosine similarity means the chunk and query are more semantically related. In practice, FAISS uses L2 distance on normalized vectors, which is equivalent to cosine similarity but more efficient computationally. The system returns the top k chunks, with k set to 5 by default, ranked from most similar to least similar.

The ranking is based only on vector similarity. There is no extra re ranking stage and no query expansion. So this is a pretty straightforward dense retrieval setup. That does mean the system can still miss some relevant chunks, especially if they are phrased in a very unusual way, but in general the embeddings do a good job capturing similar meaning even when the exact words do not match.

### Retrieval Augmented Generation (RAG)

After the top k chunks are retrieved, the system uses Retrieval Augmented Generation to answer the user’s question. The retrieved chunks are combined and passed as context into OpenAI’s GPT 4o mini model along with the original user query. This helps ground the response in the actual corpus and reduces the chances of the model just making something up from pretraining.

The system supports both single turn and multi turn interaction. In single turn mode, each question is treated independently through `ask.py`. In multi turn mode, `chat.py` keeps a conversation history using a `ConversationManager`, which stores the last 20 message pairs. That lets the system handle follow up questions and maintain context across turns.

### Technical Specifications

**Embedding Model:** OpenAI `text-embedding-3-small` (1536 dimensions)  
**Vector Store:** FAISS with exact k NN search  
**Chunk Size:** 1000 characters with 200 character overlap  
**Similarity Metric:** Cosine similarity via L2 distance on normalized vectors  
**Retrieval:** Top 5 chunks by default, configurable up to k = 10  
**Generation Model:** GPT 4o mini with temperature 0.7  
**Corpus Size:** 9 documents, 4,329 indexed chunks  
**Index Storage:** Local FAISS index on disk, approximately 5 MB  

The system was implemented in Python using OpenAI’s API for embeddings and generation, LangChain for document splitting and FAISS integration, and PyMuPDF for PDF extraction. Everything runs locally other than the OpenAI API calls.

---

## LLM response Evaluation

Overall, the LLM did an excellent job synthesizing the information that it received and providing users with proper responses; it also was able to answer follow-up questions well.



### Query Execution and Result Set

For each of the three test queries, the system retrieved the top 5 most similar chunks from the corpus using the semantic retrieval pipeline described above. I chose k = 5 for a couple reasons. First, since the retrieved chunks are passed to a language model as context, giving too many chunks can dilute the quality of the answer instead of improving it. Second, keeping retrieval to 5 chunks helps control both API cost and response time. For this kind of educational system, precision matters more than trying to retrieve everything possible. It is more useful to return 5 strong chunks than 10 mixed quality ones.

The evaluation was done on the retrieved chunks before they were passed into the language model. That way I could evaluate retrieval quality directly instead of mixing retrieval performance with generation quality.

### Relevance Assessment

After retrieving the top 5 chunks for each query, I manually reviewed each one and assigned a relevance score. The scale used was 3 for highly relevant, 2 for somewhat relevant, 1 for marginally relevant, and 0 for not relevant. Chunks with a score of 0 were treated as non relevant.

A score of 3 meant the chunk directly addressed the question and provided key information needed for the answer. A score of 2 meant the chunk was related and useful, but not central. A score of 1 meant there was some connection, but it was weak or only partly helpful. A score of 0 meant the chunk did not really help answer the query at all.

This judgment process obviously includes some subjectivity, but there is really no way around that here. If you want to know whether retrieved information actually answers a question, at some point a person has to read it and decide. For example, a chunk explaining the debt snowball method was judged highly relevant for the debt payoff query even if it did not explicitly say “credit card debt,” because it still clearly applied. On the other hand, chunks that were basically table of contents lines or page headers were marked as not relevant even if they contained matching keywords.

**Important Limitation:** Since the corpus contains 4,329 chunks, it was not realistic to manually assess every chunk for every query. That means I could not calculate true recall. Instead, I focused on Precision@5 and nDCG@5, which are both still useful for measuring the quality of the retrieved results.

### Evaluation Metrics

**Precision@5** measures how many of the 5 retrieved chunks were actually relevant. In other words, it answers the question, of the 5 returned chunks, how many were useful?

**nDCG@5** measures ranking quality while also taking graded relevance into account. It gives more credit when highly relevant chunks appear at the top, and less credit when they appear lower down. Since users usually pay the most attention to the first few results, this is a useful metric for this type of system.

---

## Evaluation Results

Overall the system performed pretty well across all three queries. Precision is high and the ranking is also strong, which is really what you want for this kind of setup.

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
- nDCG@5: 1.000  
- One chunk came from a table of contents and was not actually useful  

**Query 2: "What is the best way to create and stick to a monthly budget?"**
- Precision@5: 1.000 (all 5 retrieved chunks were relevant)  
- nDCG@5: 0.944  

**Query 3: "What strategies can I use to pay off my credit card debt?"**
- Precision@5: 1.000 (all 5 retrieved chunks were relevant)  
- nDCG@5: 0.944  

### Aggregate Performance

| Metric | Average Score |
|--------|---------------|
| **Precision@5** | **0.933** (93.3%) |
| **nDCG@5** | **0.963** (96.3%) |

These results show that the system is generally retrieving the right information. With 93.3% precision, almost every chunk that was returned was actually useful. Out of 15 total retrieved chunks across all queries, only one was clearly not relevant. That is a pretty good signal that the retrieval step is doing its job.

The nDCG score being above 0.96 also matters. It means the most relevant chunks are usually showing up near the top of the results, which is what users will actually see. Even if you retrieve good content, it does not help much if it is buried lower in the ranking.

It is also worth noting that this was done with k = 5. With such a small retrieval window, every slot matters. So getting 14 out of 15 relevant chunks is a strong result. The one failure case, where a table of contents chunk was returned, is a pretty typical issue when working with raw PDF text. It is not really a model failure as much as a preprocessing issue.

---

## Discussion

### Strengths of the Semantic Search Approach

The biggest strength of this system is its ability to match meaning instead of just matching words. Traditional keyword search would struggle with some of these queries unless the exact terms lined up. Here, the embedding model is able to connect related ideas even when the wording is different.

This shows up clearly in the debt query. The user asked about "credit card debt," but the system still retrieved chunks talking about "high-interest debt" and "unsecured debt." Those are obviously relevant, but they do not match the query exactly. A keyword system might miss those entirely.

Another strong point is how simple the pipeline is. There is no complicated query processing, no synonym dictionaries, no manual rules. The embedding model handles most of that implicitly. That makes the system easier to extend to other domains without having to redesign everything.

FAISS also works well here. Even though the current dataset is not huge, the same setup would scale to much larger corpora without major changes. That is important if this were to be turned into a real product.

The RAG layer is also doing useful work. Instead of just returning chunks, the system can generate a clean answer grounded in the retrieved content. That makes it much more usable from a user perspective. The multi-turn chat support is also nice, since users rarely ask everything in one perfect query.

---

### Limitations and Challenges

Even though the results are strong, there are still some clear limitations.

First, the system relies entirely on semantic similarity. There is no keyword or term matching component. That can be a problem in edge cases where exact phrasing matters. A hybrid approach (dense + BM25) would probably improve robustness a bit.

Second, there is no real query understanding. Everything gets embedded as a single vector. If a user asks a multi-part question, the system is not explicitly breaking it down. That could lead to missing parts of the query, especially if different parts map to different sections of the corpus.

Third, preprocessing could be improved. The table of contents chunk that showed up in Query 1 is a good example. Filtering out low-information chunks like headers or TOCs would likely improve precision even further without changing the model at all.

Finally, the system depends heavily on OpenAI models. The embeddings and generation are both external and somewhat opaque. If something goes wrong, it is hard to debug exactly why, since we cannot inspect how the embedding model is representing the text. It also means there is some dependency on API cost and availability.

---

## How to use

This project can be run locally. To get started, follow the instructions in the `USAGE.md` file in the repository.