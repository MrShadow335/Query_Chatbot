from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(file_paths: List[str]) -> List[Document]:
    """Load and split documents into chunks"""
    if not file_paths:
        raise ValueError("No file paths provided"):
    documents = []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif path.endswith(".eml"):
            loader = UnstructuredEmailLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)

    # Split into chunks for indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len,)
    return text_splitter.split_documents(documents)



