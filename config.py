"""
Configuration for Financial Planning Search Engine
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Chat Configuration
CHAT_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
MAX_RESPONSE_TOKENS = 500

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Conversation Settings
MAX_HISTORY_LENGTH = 20  # Keep last 20 message pairs

# Paths
PROJECT_ROOT = Path(__file__).parent
PDFS_DIR = PROJECT_ROOT / "pdfs"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
INDEX_DIR = PROJECT_ROOT / "faiss_index"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# Create directories if they don't exist
PDFS_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
