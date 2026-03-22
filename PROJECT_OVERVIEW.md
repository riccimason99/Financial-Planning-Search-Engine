# Project Overview - INFO 624 Final Project

## What You Have

A **complete, standalone Information Retrieval system** for financial planning:

### Location
```
/Users/riccimason99/PycharmProjects/Financial-Planning-Search-Engine/
```

### Complete Independence

**Uses from old repo:** NOTHING (100% independent)
- API key is now hardcoded in `config.py`
- No Django
- No Docker  
- No PostgreSQL
- No Redis
- Completely portable

### System Components

1. **search_engine.py** - Core FAISS search + RAG
2. **conversation_manager.py** - Multi-turn memory
3. **pdf_converter.py** - PDF → Markdown
4. **chat.py** - Interactive interface
5. **ask.py** - Single-question interface
6. **evaluate.py** - IR metrics calculator
7. **config.py** - All settings in one place

### Already Working

✅ 740 document chunks indexed  
✅ FAISS vector store built  
✅ 6 financial documents loaded (3 PDFs + 3 samples)  
✅ Semantic search working  
✅ RAG natural language responses  
✅ Conversation memory (multi-turn)  
✅ Git repository initialized  

## Workflow

### Daily Use

```bash
# Ask single questions
python3 ask.py "How do I save for college?"

# Interactive chat (with memory)
python3 chat.py
```

### Adding New Documents

```bash
# 1. Drop PDFs in pdfs/ folder
cp ~/Downloads/new-doc.pdf pdfs/

# 2. Convert
python3 pdf_converter.py pdfs -o documents

# 3. Rebuild index
python3 search_engine.py

# Done! New documents are searchable
```

## For Your Assignment

### 1. Define Use Cases

Edit `test_queries.json` with 3 real use cases:
- Use Case 1: Student needs retirement planning info
- Use Case 2: User wants debt payoff strategy
- Use Case 3: Family building emergency fund

### 2. Run Evaluation

```bash
python3 evaluate.py
```

Gets: Precision, Recall, F1, nDCG

### 3. Document Your System

Use the README.md as a template. Document:

**Why:** Help people find financial planning information  
**What:** CFPB documents + financial guides  
**Who:** 3 user personas with specific needs  
**How:** 
- Indexing: FAISS + OpenAI embeddings
- Query: Semantic search with cosine similarity
- Ranking: Vector similarity scores
- Chunk size: 1000 chars, 200 overlap

**Where:** `faiss_index/` local directory

**Evaluation:** Your metrics from evaluate.py

## Architecture Comparison

### Old Django System (Complex)
- 15+ containers running
- PostgreSQL database
- Redis cache
- 40+ function tools
- Token tracking
- Subscription management
- **Status:** Dependency issues, not working

### New IR System (Simple)
- Pure Python scripts
- File-based storage
- FAISS index
- 1 core feature: Search
- **Status:** Fully working ✅

## Next Steps

1. Delete sample docs (sample_*.md) if you want only your PDFs
2. Add more PDFs to pdfs/ folder
3. Define your 3 use cases
4. Run evaluation
5. Write report with metrics

Everything you need for the assignment is here and working!
