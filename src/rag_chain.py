"""RAG chain combining retrieval and LLM generation."""

from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS
from src.vector_store import similarity_search, hybrid_search, keyword_search


SYSTEM_PROMPT = """You are a knowledgeable research assistant with expertise in analyzing documents.

## Your Task
Answer the user's question using ONLY the provided context. Follow these guidelines:

1. **Accuracy**: Only state facts found in the context. Never invent information.
2. **Citations**: Always cite sources using [X] notation where X is the source number.
3. **Reasoning**: When synthesizing from multiple sources, explain your reasoning.
4. **Uncertainty**: If the context is incomplete or ambiguous, explicitly state what is missing.
5. **Structure**: For complex answers, use bullet points or numbered lists.

## Handling Edge Cases
- If no relevant information exists: State "The provided documents do not contain information about [topic]."
- If information is partial: Provide what is available and note the gaps.
- If sources conflict: Present both perspectives with their respective citations."""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based on the context above. If you find relevant information, cite the source number [X]."""


def get_llm() -> ChatOllama:
    """Get the Ollama LLM."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string with rich metadata."""
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        file_type = doc.metadata.get("file_type", "")
        
        # Build metadata line
        meta_parts = [f"Source: {source}"]
        if title:
            meta_parts.append(f"Title: {title}")
        if file_type:
            meta_parts.append(f"Type: {file_type}")
        meta_line = " | ".join(meta_parts)
        
        context_parts.append(f"[{i}] ({meta_line})\n{doc.page_content}")
    
    return "\n\n".join(context_parts)


SearchMode = Literal["hybrid", "vector", "keyword"]


def query_rag(
    question: str, 
    k: int = TOP_K_RESULTS,
    search_mode: SearchMode = "hybrid",
    title_filter: str | None = None
) -> tuple[str, list[Document]]:
    """
    Query the RAG system.
    
    Args:
        question: The user's question
        k: Number of documents to retrieve
        search_mode: "hybrid" (default), "vector", or "keyword"
        title_filter: Optional title substring filter (case-insensitive)
        
    Returns:
        Tuple of (answer, retrieved_documents)
    """
    # Retrieve relevant documents based on search mode
    if search_mode == "hybrid":
        documents = hybrid_search(question, k=k, title_filter=title_filter)
    elif search_mode == "keyword":
        documents = keyword_search(question, k=k, title_filter=title_filter)
    else:  # vector
        documents = similarity_search(question, k=k, title_filter=title_filter)
    
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
