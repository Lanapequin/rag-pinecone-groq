import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "langchain-rag-demo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, 384-dim
EMBEDDING_DIMENSION = 384


def get_embeddings():
    """Free embeddings via HuggingFace sentence-transformers."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    """Groq LLM — free tier with llama3."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def init_pinecone_index():
    """Initialize Pinecone and create index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    existing = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully!")
    else:
        print(f"Using existing Pinecone index: {INDEX_NAME}")
    
    return pc.Index(INDEX_NAME)

def ingest_documents(texts: list[dict]) -> PineconeVectorStore:
    """
    Ingest documents into Pinecone.
    
    Args:
        texts: List of dicts with 'content' and 'metadata' keys
    
    Returns:
        PineconeVectorStore instance
    """
    from langchain_core.documents import Document
    
    print(f"\nIngesting {len(texts)} documents into Pinecone...")
    
    docs = [
        Document(page_content=t["content"], metadata=t.get("metadata", {}))
        for t in texts
    ]
    
    embeddings = get_embeddings()
    
    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    print("Documents ingested successfully!")
    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """Load an existing Pinecone vector store."""
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vector_store: PineconeVectorStore):
    """
    Build the RAG chain.
    
    Flow: Question → Retriever → Context + Question → LLM → Answer
    """

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question 
based ONLY on the provided context. If the answer is not in the context, 
say "I don't have enough information to answer that question."

Context:
{context}"""),
        ("human", "{question}")
    ])
    
    llm = get_llm()
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def ask_with_sources(question: str, retriever, rag_chain) -> dict:
    """Ask a question and return both the answer and source documents."""
    
    source_docs = retriever.invoke(question)
    
    answer = rag_chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in source_docs
        ]
    }


# Sample Data

SAMPLE_DOCUMENTS = [
    {
        "content": """LangChain is an open-source framework designed to simplify the creation of 
applications using large language models (LLMs). It provides a standard interface for chains, 
lots of integrations with other tools, and end-to-end chains for common applications. LangChain 
was launched in October 2022 by Harrison Chase while working at Robust Intelligence.""",
        "metadata": {"source": "LangChain Overview", "topic": "langchain"}
    },
    {
        "content": """Retrieval-Augmented Generation (RAG) is an AI framework that enhances the accuracy 
and reliability of generative AI models by incorporating facts fetched from external sources. RAG 
combines the strengths of retrieval-based models with generative models. The retrieval component 
searches a knowledge base for relevant information, while the generation component uses this 
information to produce accurate, contextually relevant responses.""",
        "metadata": {"source": "RAG Explained", "topic": "rag"}
    },
    {
        "content": """Pinecone is a fully managed vector database that makes it easy to add vector 
search to production applications. Vector databases store high-dimensional vectors (embeddings) 
and enable fast similarity search. Pinecone offers a free tier called "Starter" with 1 project, 
1 index, and 2GB storage — perfect for learning and small projects.""",
        "metadata": {"source": "Pinecone Documentation", "topic": "pinecone"}
    },
    {
        "content": """Groq is a company that provides inference infrastructure for large language models. 
Groq's Language Processing Unit (LPU) offers significantly faster inference speeds compared to 
traditional GPU-based solutions. Groq offers a free API tier called GroqCloud that developers can 
use to build applications with models like LLaMA 3, Mixtral, and Gemma.""",
        "metadata": {"source": "Groq Documentation", "topic": "groq"}
    },
    {
        "content": """Vector embeddings are numerical representations of data (text, images, etc.) in 
a high-dimensional space. Similar concepts are represented by vectors that are close together in 
this space. In NLP, embeddings capture semantic meaning — words or sentences with similar meanings 
have embeddings that are close in vector space. Common embedding models include OpenAI's 
text-embedding-ada-002 and open-source models like all-MiniLM-L6-v2 from Sentence Transformers.""",
        "metadata": {"source": "Embeddings Guide", "topic": "embeddings"}
    },
    {
        "content": """LangChain Expression Language (LCEL) is a declarative way to compose chains in 
LangChain. It uses the pipe operator (|) to chain components together. LCEL supports streaming, 
async operations, parallel execution, and easy debugging. A simple chain looks like: 
chain = prompt | llm | output_parser. This makes it easy to read and understand the data flow.""",
        "metadata": {"source": "LCEL Documentation", "topic": "lcel"}
    },
    {
        "content": """HuggingFace is a platform and AI community that offers thousands of pre-trained 
models for various NLP tasks. The sentence-transformers library, available through HuggingFace, 
provides models specifically designed for generating high-quality sentence embeddings. The model 
all-MiniLM-L6-v2 is particularly popular — it's fast, small (80MB), produces 384-dimensional 
embeddings, and can be used completely free without any API key.""",
        "metadata": {"source": "HuggingFace Guide", "topic": "huggingface"}
    },
]


# ─── Main Demo ────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("    RAG SYSTEM WITH PINECONE + GROQ + HUGGINGFACE EMBEDDINGS")
    print("=" * 65)
    
    # Step 1: Initialize Pinecone
    print("\nStep 1: Initializing Pinecone...")
    init_pinecone_index()
    
    # Step 2: Ingest documents
    print("\nStep 2: Loading and ingesting documents...")
    vector_store = ingest_documents(SAMPLE_DOCUMENTS)
    
    # Step 3: Build RAG chain
    print("\nStep 3: Building RAG chain...")
    rag_chain, retriever = build_rag_chain(vector_store)
    print("RAG chain ready!")
    
    # Step 4: Ask questions
    questions = [
        "What is LangChain and when was it created?",
        "How does RAG work and what are its benefits?",
        "What is Pinecone and is there a free tier?",
        "How can I use embeddings for free without an API key?",
        "What is LCEL and how do you use the pipe operator?",
    ]
    
    print("\n" + "=" * 65)
    print("    QUESTION & ANSWER DEMO")
    print("=" * 65)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)
        
        result = ask_with_sources(question, retriever, rag_chain)
        
        print(f"Answer: {result['answer']}")
        print(f"\nSources used:")
        for src in result["sources"]:
            print(f"   • {src['metadata'].get('source', 'Unknown')}")
        print()
    
    print("=" * 65)
    print("RAG Demo completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
