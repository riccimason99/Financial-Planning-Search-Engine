"""
Financial Planning Search Engine - Information Retrieval Final Project

This module implements a semantic search engine using FAISS vector similarity.
It demonstrates key IR concepts: indexing, embedding-based retrieval, and ranking.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import config


class FinancialSearchEngine:
    """
    A semantic search engine for financial planning documents.
    
    Uses OpenAI embeddings + FAISS for efficient similarity search.
    """
    
    def __init__(self, openai_api_key: str = None, index_path: str = None):
        """
        Initialize the search engine.
        
        Args:
            openai_api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
            index_path: Path to save/load the FAISS index (defaults to config.INDEX_DIR)
        """
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.index_path = index_path or str(config.INDEX_DIR)
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=self.openai_api_key
        )
        self.client = OpenAI(api_key=self.openai_api_key)
        self.vector_store = None
        self.documents = []
        
    def load_documents_from_directory(self, directory: str) -> int:
        """
        Load all text/markdown files from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Number of document chunks created
        """
        print(f"Loading documents from {directory}...")
        
        all_text = []
        file_sources = []
        
        doc_dir = Path(directory)
        for file_path in doc_dir.glob("**/*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                all_text.append(content)
                file_sources.append(str(file_path.name))
        
        for file_path in doc_dir.glob("**/*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                all_text.append(content)
                file_sources.append(str(file_path.name))
        
        if not all_text:
            print(f"No documents found in {directory}")
            return 0
        
        combined_text = "\n\n---\n\n".join(
            f"Source: {source}\n{text}" 
            for source, text in zip(file_sources, all_text)
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(combined_text)
        self.documents = chunks
        
        print(f"Created {len(chunks)} document chunks from {len(all_text)} files")
        return len(chunks)
    
    def build_index(self) -> bool:
        """
        Build FAISS index from loaded documents.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.documents:
            print("No documents loaded. Call load_documents_from_directory() first.")
            return False
        
        print(f"Building FAISS index with {len(self.documents)} chunks...")
        print("Generating embeddings (this may take a minute)...")
        
        self.vector_store = FAISS.from_texts(
            texts=self.documents,
            embedding=self.embeddings
        )
        
        print(f"✓ Index built successfully with {len(self.documents)} vectors")
        return True
    
    def save_index(self) -> bool:
        """Save FAISS index to disk."""
        if not self.vector_store:
            print("No index to save. Build index first.")
            return False
        
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)
        print(f"✓ Index saved to {self.index_path}")
        return True
    
    def load_index(self) -> bool:
        """Load FAISS index from disk."""
        if not os.path.exists(self.index_path):
            print(f"Index not found at {self.index_path}")
            return False
        
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Index loaded from {self.index_path}")
        return True
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (document_text, similarity_score) tuples
        """
        if not self.vector_store:
            print("No index loaded. Build or load index first.")
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [(doc.page_content, score) for doc, score in results]
    
    def ask(self, question: str, k: int = 5, model: str = None, 
            conversation_manager=None) -> str:
        """
        Ask a question and get a natural language response using RAG.
        
        This is the full Retrieval-Augmented Generation flow:
        1. Search for relevant documents
        2. Pass them as context to OpenAI
        3. Get natural language response
        
        Args:
            question: The user's question
            k: Number of document chunks to retrieve
            model: OpenAI model to use for response generation
            
        Returns:
            Natural language answer from the LLM
        """
        if not self.vector_store:
            return "Error: No index loaded. Build or load index first."
        
        # Step 1: Retrieve relevant documents
        search_results = self.search(question, k=k)
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Step 2: Format context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(search_results, 1):
            context_parts.append(f"[Document {i}]\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Step 3: Build messages for OpenAI
        if conversation_manager:
            # Multi-turn conversation with memory
            conversation_manager.add_user_message(question)
            
            # Add retrieved context as a system message
            context_message = f"""Here are relevant document excerpts to help answer:

{context}

Use this information to provide an accurate answer. Cite documents when relevant."""
            
            # Get full conversation history
            messages = conversation_manager.get_messages()
            
            # Insert context before the last user message
            messages_with_context = messages[:-1] + [
                {"role": "system", "content": context_message}
            ] + [messages[-1]]
            
        else:
            # Single-turn (no memory)
            system_prompt = """You are a helpful financial planning assistant. 
Use the provided document excerpts to answer the user's question accurately.
If the documents don't contain enough information, say so.
Cite which documents you're using (e.g., "According to Document 1...").
Keep your answer clear and concise."""
            
            user_prompt = f"""Based on these document excerpts:

{context}

Question: {question}

Please provide a helpful answer based on the information above."""
            
            messages_with_context = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        # Step 4: Call OpenAI for natural language response
        try:
            response = self.client.chat.completions.create(
                model=model or config.CHAT_MODEL,
                messages=messages_with_context,
                temperature=config.CHAT_TEMPERATURE,
                max_tokens=config.MAX_RESPONSE_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Save assistant response to conversation history
            if conversation_manager:
                conversation_manager.add_assistant_message(answer)
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    """Demo usage of the search engine."""
    import sys
    
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        print("Error: OPENAI_API_KEY not set")
        print("\nEdit config.py and set your API key, or:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)
    
    engine = FinancialSearchEngine()
    
    print("\n" + "="*60)
    print("Financial Planning Search Engine - Demo")
    print("="*60 + "\n")
    
    docs_dir = str(config.DOCUMENTS_DIR)
    
    num_chunks = engine.load_documents_from_directory(docs_dir)
    
    if num_chunks == 0:
        print("\nNo documents found. Add .md or .txt files to ./documents/")
        return
    
    if not engine.build_index():
        return
    
    engine.save_index()
    
    print("\n" + "="*60)
    print("Testing Search")
    print("="*60 + "\n")
    
    test_queries = [
        "How should I save for retirement?",
        "What's the best strategy to pay off debt?",
        "How do I build an emergency fund?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        results = engine.search(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            preview = doc[:200].replace("\n", " ")
            print(f"{i}. Score: {score:.4f}")
            print(f"   {preview}...")
            print()


if __name__ == "__main__":
    main()
