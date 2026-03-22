"""
Conversation History Manager - Adapted from existing chatbot code

Manages multi-turn conversations with memory, based on the ChatHistoryManager
from backend/apps/chatbot/utils.py but simplified to work without Redis/Django.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

import config


class ConversationManager:
    """
    Handles conversation history for multi-turn interactions.
    
    Adapted from the existing ChatHistoryManager but uses JSON files instead of Redis.
    """
    
    def __init__(self, session_id: str = "default", history_length: int = None, 
                 sessions_dir: str = None):
        """
        Initialize conversation manager.
        
        Args:
            session_id: Unique identifier for this conversation
            history_length: Max number of message pairs to keep
            sessions_dir: Directory to store conversation files
        """
        self.session_id = session_id
        self.history_length = history_length or config.MAX_HISTORY_LENGTH
        self.sessions_dir = Path(sessions_dir) if sessions_dir else config.SESSIONS_DIR
        self.sessions_dir.mkdir(exist_ok=True)
        
        self.session_file = self.sessions_dir / f"{session_id}.json"
        self.chat_history = self._load_chat_history()
    
    def _load_chat_history(self) -> List[Dict]:
        """Load chat history from file, initialize if needed."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    history = json.load(f)
                    print(f"✓ Loaded conversation with {len(history)-1} messages")
                    return history
            except Exception as e:
                print(f"Warning: Could not load history: {e}")
        
        # Initialize with system prompt
        system_prompt = """You are a helpful financial planning assistant.
You have access to financial planning documents and can answer questions about:
- Retirement savings strategies
- Budgeting and spending tracking
- Debt payoff methods
- Emergency funds
- Investment strategies
- Financial education

Provide clear, accurate answers based on the retrieved documents.
When relevant information is in the documents, cite them.
If information is not available, say so honestly."""
        
        history = [{"role": "system", "content": system_prompt}]
        self._save_chat_history(history)
        return history
    
    def _save_chat_history(self, history: List[Dict] = None):
        """Save chat history to file."""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(history or self.chat_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def _truncate_history(self):
        """Keep only recent messages (system prompt + last N messages)."""
        if len(self.chat_history) > self.history_length + 1:
            # Keep system prompt + last N messages
            self.chat_history = [
                self.chat_history[0]  # System prompt
            ] + self.chat_history[-self.history_length:]
    
    def add_user_message(self, message: str):
        """Add user message to history."""
        self.chat_history.append({"role": "user", "content": message})
        self._truncate_history()
        self._save_chat_history()
    
    def add_assistant_message(self, message: str):
        """Add assistant message to history."""
        self.chat_history.append({"role": "assistant", "content": message})
        self._save_chat_history()
    
    def get_messages(self) -> List[Dict]:
        """Get all messages for sending to OpenAI."""
        return self.chat_history
    
    def clear_history(self):
        """Reset conversation to initial state."""
        system_prompt = self.chat_history[0]["content"]
        self.chat_history = [{"role": "system", "content": system_prompt}]
        self._save_chat_history()
        print("✓ Conversation history cleared")
    
    def delete_session(self):
        """Delete the session file."""
        if self.session_file.exists():
            self.session_file.unlink()
            print(f"✓ Session {self.session_id} deleted")
