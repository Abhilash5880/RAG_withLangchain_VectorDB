import tempfile
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

def create_kb(pdf_path: str = "langchain_demo.pdf"):
    """Create a vector store from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # Returns a list of Document objects (one per page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} #question is passed as-is without changing
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    # Test
    questions = [
        "List all 8 document loaders mentioned in the PDF and explain what each one does.?",
        "What are all 5 best practices for document loaders, and what are all 6 key features of LangChain listed in the PDF?",
        "In the code example on Page 1, what file path is passed to PyPDFLoader, and how is the page printed?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    demo_basic_rag()