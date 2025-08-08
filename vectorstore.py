import getpass
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus


# Setup Google API Key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize Gemini Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document"  # Optimized for document retrieval
)

# Create temporary database file
db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

# Initialize Milvus Vector Store
vector_db = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": db_file},
    collection_name="RAG_Collection",
    auto_id=True,
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    drop_old=False
)


def setup_vectorstore():
    """Initialize and populate the vector store"""
    # Load and split documents
    file_paths = ["insurance_policy_v3.pdf", "exclusions_clause.docx"]
    docs = load_and_split_documents(file_paths)
    
    print(f"Loaded {len(docs)} document chunks")
    
    # Add documents to Milvus
    vector_db.add_documents(docs)
    print("Documents indexed successfully!")
    
    return vector_db

def query_vectorstore(query: str, k: int = 5):
    """Query the vector store for similar documents"""
    results = vector_db.similarity_search(query, k=k)
    return results

# Initialize the vectorstore
if __name__ == "__main__":
    vectorstore = setup_vectorstore()
    
    # Example query
    test_query = "What are the policy exclusions?"
    results = query_vectorstore(test_query)
    
    print(f"\nQuery: {test_query}")
    print("Results:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:200]}...")

