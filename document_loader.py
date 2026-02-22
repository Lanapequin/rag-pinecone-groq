import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_file(filepath: str) -> List[Document]:
    """Load a plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return [Document(
        page_content=content,
        metadata={"source": filepath, "type": "text"}
    )]


def load_from_url(url: str) -> List[Document]:
    """Load content from a web URL."""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs
    except ImportError:
        print("Install: pip install langchain-community beautifulsoup4")
        return []


def load_pdf(filepath: str) -> List[Document]:
    """Load a PDF file."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        return pages
    except ImportError:
        print("Install: pip install langchain-community pypdf")
        return []


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of smaller Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def create_sample_knowledge_base(output_dir: str = "knowledge_base"):
    """Create sample text files to use as knowledge base."""
    os.makedirs(output_dir, exist_ok=True)
    
    articles = {
        "ai_history.txt": """
Artificial Intelligence: A Brief History

The field of artificial intelligence (AI) was founded at the Dartmouth Conference in 1956. 
Researchers John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester organized 
the first AI conference, coining the term "artificial intelligence."

Early AI systems used rule-based approaches and symbolic logic. The first AI programs 
included the Logic Theorist (1956) and General Problem Solver (1957) by Allen Newell and 
Herbert Simon.

The AI winter of the 1970s-1980s saw reduced funding and interest due to overpromised results.
However, the field revived with expert systems in the 1980s, machine learning in the 1990s,
and deep learning revolution starting in 2012.

Modern AI milestones include:
- 1997: Deep Blue defeats chess champion Garry Kasparov
- 2011: IBM Watson wins Jeopardy!
- 2012: AlexNet wins ImageNet competition, launching deep learning era
- 2017: Attention Is All You Need paper introduces the Transformer architecture
- 2022: ChatGPT launches, making LLMs mainstream
- 2023: GPT-4, Claude, and Gemini compete as frontier AI models
        """,
        
        "llm_basics.txt": """
Large Language Models (LLMs): Core Concepts

A Large Language Model (LLM) is a type of AI model trained on vast amounts of text data 
to understand and generate human language. LLMs use the Transformer architecture, introduced 
in the "Attention Is All You Need" paper by Vaswani et al. in 2017.

Key characteristics of LLMs:
1. Scale: Trained on billions or trillions of tokens
2. Emergent capabilities: Abilities that appear at sufficient scale
3. Few-shot learning: Can learn new tasks from just a few examples
4. In-context learning: Uses examples in the prompt

Popular LLMs include:
- GPT-4 by OpenAI (closed source)
- Claude by Anthropic (closed source)
- LLaMA 3 by Meta (open source)
- Gemini by Google (closed source)
- Mixtral by Mistral AI (open weights)

Training process involves:
1. Pre-training on large text corpora
2. Fine-tuning with supervised learning
3. RLHF (Reinforcement Learning from Human Feedback)
        """,
        
        "vector_databases.txt": """
Vector Databases: Complete Guide

A vector database is a specialized database designed to store, index, and query 
high-dimensional vectors (embeddings) efficiently. Unlike traditional databases 
that search for exact matches, vector databases perform similarity search.

How vector search works:
1. Data (text, images) is converted to embeddings using an embedding model
2. Embeddings are stored in the vector database
3. At query time, the query is also converted to an embedding
4. The database finds stored vectors most similar to the query vector
5. Similarity is measured using cosine similarity, dot product, or Euclidean distance

Popular vector databases:
- Pinecone: Fully managed, easy to use, generous free tier
- Weaviate: Open-source, can run locally
- Chroma: Lightweight, great for development
- Qdrant: Open-source, high performance
- Milvus: Open-source, enterprise-grade
- pgvector: PostgreSQL extension for vector storage

Pinecone Free Tier (Starter Plan):
- 1 project
- 1 index
- Up to 2GB storage
- Shared infrastructure
- Perfect for learning and development
        """
    }
    
    for filename, content in articles.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Created: {filepath}")
    
    return output_dir


if __name__ == "__main__":
    print("Creating sample knowledge base...")
    kb_dir = create_sample_knowledge_base()
    
    print("\nLoading and splitting documents...")
    all_docs = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            docs = load_text_file(filepath)
            all_docs.extend(docs)
    
    chunks = split_documents(all_docs, chunk_size=300, chunk_overlap=50)
    print(f"\nReady to ingest {len(chunks)} chunks into Pinecone")
