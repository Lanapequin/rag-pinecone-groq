from rag_app import (
    init_pinecone_index,
    load_vector_store,
    build_rag_chain,
    ask_with_sources,
    ingest_documents,
    SAMPLE_DOCUMENTS
)


def print_banner():
    print("\n" + "═" * 60)
    print("RAG CHAT - Powered by Pinecone + Groq + HuggingFace")
    print("═" * 60)
    print("  Commands:")
    print("    Type your question and press Enter")
    print("    'sources' - Show sources from last answer")
    print("    'reload'  - Re-ingest sample documents")
    print("    'quit'    - Exit the chatbot")
    print("═" * 60 + "\n")


def main():
    print_banner()
    
    print("Initializing system...")
    init_pinecone_index()
    
    print("Loading documents into vector store...")
    vector_store = ingest_documents(SAMPLE_DOCUMENTS)
    
    print("Building RAG chain...")
    rag_chain, retriever = build_rag_chain(vector_store)
    
    print("\nSystem ready! Ask me anything about LangChain, RAG, Pinecone, Groq, or Embeddings.\n")
    
    last_result = None
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        elif user_input.lower() == "sources":
            if last_result:
                print("\nSources from last answer:")
                for i, src in enumerate(last_result["sources"], 1):
                    print(f"  {i}. [{src['metadata'].get('source', 'Unknown')}]")
                    print(f"     {src['content'][:150]}...")
                print()
            else:
                print("No previous answer to show sources for.\n")
        
        elif user_input.lower() == "reload":
            print("Reloading documents...")
            vector_store = ingest_documents(SAMPLE_DOCUMENTS)
            rag_chain, retriever = build_rag_chain(vector_store)
            print("Documents reloaded!\n")
        
        else:
            print("\nThinking...")
            last_result = ask_with_sources(user_input, retriever, rag_chain)
            print(f"\nAssistant: {last_result['answer']}")
            print(f"\n   [Sources: {', '.join(s['metadata'].get('source', '?') for s in last_result['sources'])}]")
            print()


if __name__ == "__main__":
    main()
