from fastapi import FastAPI, File, UploadFile, HTTPException
from main import query_rag_system
from query_parser import parse_insurance_query
from retriever import query_rag_system, get_retrieval_context
from decision_engine import process_claim_decision, get_decision_summary
from typing import List, Dict, Any
import uvicorn

app = FastAPI()

@app.post("/query")
async def query_endpoint(question: str):
    # Parse query for structured data
    parsed = parse_insurance_query(question)
    
    # Get RAG answer
    result = query_rag_system(question)
    
    return {
        "answer": result,
        "parsed_query": parsed,
        "search_strategy": parsed["enhanced_search_phrases"]
    }

@app.post("/process_query/")
async def process_query(
    query: str,
    documents: List[UploadFile]  
) -> Dict[str, Any]:
    """
    Process a natural language query and documents, return a structured response.
    """
    structured_query = parse_query(query)  

    full_texts = [await extract_text(file) for file in documents] 

    relevant_clauses = find_relevant_clauses(structured_query, full_texts)  
    
    result = get_retrieval_context(question)
    decision, amount, justification, clause_mapping = evaluate_logic(structured_query, relevant_clauses)  

    return {
        "decision": decision,
        "amount": amount,
        "justification": justification,
        "clause_mapping": clause_mapping
    }


@app.post("/claim-decision")
async def claim_decision_endpoint(query: str, patient_data: Optional[Dict] = None):
    decision = process_claim_decision(query, patient_data)
    summary = get_decision_summary(query, patient_data)
    
    return {
        "decision": decision,
        "summary": summary,
        "status": "success"
    }

@app.post("/query-with-decision")
async def comprehensive_query(query: str):
    # Regular RAG + Decision if applicable
    result = process_insurance_query(query)
    return result

# --------- Utility Functions to be implemented ----------

def parse_query(query: str) -> Dict[str, Any]:
    pass

async def extract_text(file: UploadFile) -> str:
    pass

def find_relevant_clauses(query_details: Dict, texts: List[str]) -> List[str]:
    pass

def evaluate_logic(query_details, clauses):
    pass

# ------------- Run Server -------------
if __name__ == "__main__":
    uvicorn.run("your_script:app", host="0.0.0.0", port=8000)




