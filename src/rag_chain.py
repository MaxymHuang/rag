"""RAG chain combining retrieval and LLM generation."""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS
from src.vector_store import similarity_search


SYSTEM_PROMPT = """You are an expert assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer is not in the context, say so.
Be concise and accurate in your responses."""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based on the context above:"""


def get_llm() -> ChatOllama:
    """Get the Ollama LLM."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    
    return "\n\n".join(context_parts)


def query_rag(question: str, k: int = TOP_K_RESULTS) -> tuple[str, list[Document]]:
    """
    Query the RAG system.
    
    Args:
        question: The user's question
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, retrieved_documents)
    """
    # Retrieve relevant documents
    documents = similarity_search(question, k=k)
    
    # Format context
    context = format_context(documents)
    
    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", RAG_PROMPT_TEMPLATE)
    ])
    
    # Get LLM and generate response
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response.content, documents
