from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from Backend_vectorstore import vector_db  # Import your vector store

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Updated to latest model
    temperature=0.2,
    convert_system_message_to_human=True
)

# Create retriever with optimized parameters
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7  # Filter low-relevance results
    }
)

# Define prompt template for document synthesis
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful insurance policy assistant. Answer the question based only on the provided context.

Context:
{context}

Question: {input}

Provide a clear, accurate answer based on the policy documents. If the information isn't in the context, say so.

Answer:""")

# Create document combination chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template
)

# Create complete RAG chain
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

def retrieve_clauses(query: str) -> list:
    """Retrieve relevant document chunks for a query"""
    relevant_docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in relevant_docs]

def query_rag_system(query: str) -> str:
    """Complete RAG query processing"""
    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return "Sorry, I couldn't process your query. Please try again."

def get_retrieval_context(query: str) -> dict:
    """Get both answer and source documents"""
    response = rag_chain.invoke({"input": query})
    return {
        "answer": response["answer"],
        "source_documents": [doc.page_content for doc in response["context"]],
        "query": query
    }

# Example usage
if __name__ == "__main__":
    test_query = "What are the policy exclusions for pre-existing conditions?"
    
    # Test retrieval
    relevant_docs = retrieve_clauses(test_query)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # Test full RAG
    answer = query_rag_system(test_query)
    print(f"Answer: {answer}")



