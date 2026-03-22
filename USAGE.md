# How to Use the Financial Planning Search Engine

## Setup (First Time Only)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your OpenAI API Key

Create a `.env` file in the project root:
```bash
echo 'OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE' > .env
```

Or use environment variable:
```bash
export OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"
```

**Note:** Never commit your `.env` file to git (it's already in `.gitignore`)

### 3. Add Documents
Place PDF files in the `pdfs/` folder, then convert them:
```bash
python3 pdf_converter.py pdfs -o documents
```

### 4. Build the Search Index
```bash
python3 search_engine.py
```

This creates the FAISS vector index from your documents.

---

## Daily Use

### Start the Interactive Chat
```bash
python3 chat.py
```

### Example Session
```
💬 You: What is a 401k?
🤖 Assistant: A 401(k) is an employer-sponsored retirement plan...

💬 You: What are the contribution limits for it?
🤖 Assistant: For 2024, the contribution limit is $23,000...

💬 You: quit
```

**Commands:**
- Type your question and press Enter
- Type `quit` or `exit` to end the session
- Type `clear` to reset conversation memory

---

## Updating Documents

When you add new PDFs:
```bash
# 1. Convert new PDFs
python3 pdf_converter.py pdfs -o documents

# 2. Rebuild index
python3 search_engine.py

# 3. Start chatting
python3 chat.py
```

---

## Evaluation

To run the evaluation metrics:
```bash
python3 evaluate.py
```

Make sure `test_queries.json` has relevance judgments filled in first.
