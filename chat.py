#!/usr/bin/env python3
"""
Interactive Chat Interface for Financial Search Engine

Multi-turn conversations with memory using RAG.
"""

import datetime
import sys

import config
from search_engine import FinancialSearchEngine
from conversation_manager import ConversationManager


def main():
    """Interactive chat with conversation memory."""
    
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        print("❌ Error: OPENAI_API_KEY not set")
        print("\nOption 1: Edit config.py and set OPENAI_API_KEY")
        print('Option 2: export OPENAI_API_KEY="sk-..."')
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Financial Planning Chatbot - RAG with Conversation Memory")
    print("="*70)
    print("\nLoading search index...")
    
    engine = FinancialSearchEngine()
    
    if not engine.load_index():
        print("\n❌ No index found. Build it first:")
        print("   python3 search_engine.py")
        sys.exit(1)
    
    print("✓ Index loaded successfully")
    
    # Initialize conversation with timestamp session ID
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation = ConversationManager(session_id=session_id)
    
    print(f"✓ Conversation memory enabled (session: {session_id})")
    print("\nCommands: 'quit' to exit, 'clear' to reset conversation\n")
    
    while True:
        try:
            question = input("💬 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'clear':
                conversation.clear_history()
                print("✓ Conversation cleared\n")
                continue
            
            print("\n🔍 Searching...", end=" ", flush=True)
            
            answer = engine.ask(
                question, 
                k=5, 
                model=config.CHAT_MODEL,
                conversation_manager=conversation
            )
            
            print("\r              \r")
            print(f"🤖 Assistant:\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
