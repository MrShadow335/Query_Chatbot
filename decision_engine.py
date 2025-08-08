from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve_clauses  # Import from your updated retriever
from query_parser import parse_insurance_query  # Import query parser
import json
import re
from typing import Dict, Optional

# Initialize Gemini LLM (consistent with your architecture)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,  # Lower temperature for consistent decisions
    convert_system_message_to_human=True
)

class InsuranceDecisionEngine:
    """
    Insurance claims adjudication system using RAG-retrieved policy clauses
    """
    
    def __init__(self):
        self.decision_prompt = PromptTemplate.from_template("""
You are an expert insurance claims adjudication system.

Based on the policy clauses and patient details, determine:
- Whether the claim should be APPROVED or REJECTED  
- If approved, suggest a payout amount (default ₹50,000 for standard surgery)
- Provide justification citing exact phrases from the clauses

### Policy Clauses:
{clauses}

### Patient Details:
- Age: {age}
- Gender: {gender}  
- Procedure: {procedure}
- Location: {location}
- Policy Duration (months): {duration}

### Decision Rules:
- Policies under 24 months do NOT cover knee/joint replacement surgeries
- Only emergency orthopedic care is allowed before 24 months
- Non-emergency knee surgery requires ≥24 months of coverage
- Pre-existing conditions may affect coverage
- Location-based treatment costs vary

Return response in strict JSON format:
{{
    "decision": "APPROVED" or "REJECTED",
    "amount": number or null,
    "justification": "detailed explanation with clause references",
    "risk_factors": ["list of identified risk factors"],
    "coverage_status": "full/partial/none"
}}

JSON Response:""")
        
        self.decision_chain = LLMChain(llm=llm, prompt=self.decision_prompt)
    
    def make_decision(self, query: str, patient_data: Optional[Dict] = None) -> Dict:
        """
        Make insurance claim decision based on query and patient data
        """
        try:
            # Parse query to extract patient details if not provided
            if not patient_data:
                parsed_query = parse_insurance_query(query)
                patient_data = {
                    "age": parsed_query.get("age", "Unknown"),
                    "gender": parsed_query.get("gender", "Unknown"),
                    "procedure": parsed_query.get("procedure", "Unknown"),
                    "location": parsed_query.get("location", "Unknown"),
                    "duration": parsed_query.get("policy_duration_months", "Unknown")
                }
            
            # Retrieve relevant policy clauses
            clauses_text = "\n\n".join(retrieve_clauses(query))
            
            # Generate decision
            decision_result = self.decision_chain.run(
                clauses=clauses_text,
                age=patient_data["age"],
                gender=patient_data["gender"],
                procedure=patient_data["procedure"],
                location=patient_data["location"],
                duration=patient_data["duration"]
            )
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', decision_result, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
            else:
                # Fallback response
                decision_data = {
                    "decision": "REJECTED",
                    "amount": None,
                    "justification": "Unable to process claim - insufficient policy information",
                    "risk_factors": ["Processing error"],
                    "coverage_status": "none"
                }
            
            # Add metadata
            decision_data["patient_details"] = patient_data
            decision_data["query"] = query
            decision_data["clauses_count"] = len(retrieve_clauses(query))
            
            return decision_data
            
        except Exception as e:
            print(f"Error in decision making: {e}")
            return {
                "decision": "REJECTED",
                "amount": None,
                "justification": f"System error: {str(e)}",
                "risk_factors": ["System error"],
                "coverage_status": "none",
                "patient_details": patient_data or {},
                "query": query,
                "error": str(e)
            }
    
    def batch_decisions(self, queries: list) -> list:
        """Process multiple claims at once"""
        results = []
        for query in queries:
            decision = self.make_decision(query)
            results.append(decision)
        return results
    
    def get_decision_summary(self, decision: Dict) -> str:
        """Generate human-readable decision summary"""
        if decision["decision"] == "APPROVED":
            return f"✅ CLAIM APPROVED - Amount: ₹{decision['amount']:,} | {decision['justification']}"
        else:
            return f"❌ CLAIM REJECTED - {decision['justification']}"

# Initialize decision engine
decision_engine = InsuranceDecisionEngine()

# Utility functions for integration
def process_claim_decision(query: str, patient_data: Optional[Dict] = None) -> Dict:
    """Main function to process insurance claim decisions"""
    return decision_engine.make_decision(query, patient_data)

def get_decision_summary(query: str, patient_data: Optional[Dict] = None) -> str:
    """Get human-readable decision summary"""
    decision = process_claim_decision(query, patient_data)
    return decision_engine.get_decision_summary(decision)

# Example usage
if __name__ == "__main__":
    # Test case from your original file
    test_query = "46-year-old male needs knee surgery in Pune with 3 months policy duration"
    
    # Using extracted patient data
    patient_info = {
        "age": 46,
        "gender": "M",
        "procedure": "knee surgery",
        "location": "Pune", 
        "duration": 3
    }
    
    decision = process_claim_decision(test_query, patient_info)
    print("Decision Result:")
    print(json.dumps(decision, indent=2))
    
    print(f"\nSummary: {get_decision_summary(test_query, patient_info)}")
