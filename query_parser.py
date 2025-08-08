from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
from typing import Dict, Optional


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

class QueryParser:
    """Parse and structure healthcare insurance queries for RAG retrieval"""
    
    def __init__(self):
        # Prompt to structure the query
        self.structure_prompt = PromptTemplate.from_template("""
You are an expert healthcare insurance query analyzer. Extract structured information from the following query.

Return ONLY valid JSON format with these fields:
- age: integer or null if not mentioned
- gender: "M", "F", or null if not mentioned  
- procedure: string describing medical procedure or null
- location: string for city/state or null
- policy_duration_months: integer or null
- query_type: "coverage", "exclusion", "claim", "premium", or "general"
- keywords: array of important terms for document retrieval

Query: {query}

JSON Output:""")
        
        self.structure_chain = LLMChain(llm=llm, prompt=self.structure_prompt)
        
        # Query enhancement for better retrieval
        self.enhancement_prompt = PromptTemplate.from_template("""
Based on this structured insurance query data, generate 2-3 alternative search phrases that would help find relevant policy documents.

Original Query: {original_query}
Structured Data: {structured_data}

Generate search phrases focusing on:
1. Policy coverage terms
2. Medical procedure terminology  
3. Exclusion clauses

Search Phrases:""")
        
        self.enhancement_chain = LLMChain(llm=llm, prompt=self.enhancement_prompt)

    def parse_query(self, query: str) -> Dict:
        """Parse query into structured format"""
        try:
            # Get structured output
            structured_output = self.structure_chain.run(query=query)
            
            # Clean and parse JSON
            json_match = re.search(r'\{.*\}', structured_output, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                # Fallback structure
                structured_data = {
                    "age": None,
                    "gender": None,
                    "procedure": None,
                    "location": None,
                    "policy_duration_months": None,
                    "query_type": "general",
                    "keywords": [query]
                }
            
            # Add original query
            structured_data["original_query"] = query
            
            return structured_data
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            return {
                "original_query": query,
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration_months": None,
                "query_type": "general",
                "keywords": [query]
            }

    def enhance_for_retrieval(self, parsed_query: Dict) -> list:
        """Generate enhanced search phrases for better document retrieval"""
        try:
            enhanced_output = self.enhancement_chain.run(
                original_query=parsed_query["original_query"],
                structured_data=json.dumps(parsed_query, indent=2)
            )
            
            # Extract search phrases
            phrases = [phrase.strip() for phrase in enhanced_output.split('\n') if phrase.strip()]
            
            # Combine with original keywords
            all_phrases = parsed_query.get("keywords", []) + phrases
            
            return list(set(all_phrases))  # Remove duplicates
            
        except Exception as e:
            print(f"Error enhancing query: {e}")
            return parsed_query.get("keywords", [parsed_query["original_query"]])

    def process_query(self, query: str) -> Dict:
        """Complete query processing pipeline"""
        # Parse query
        structured_data = self.parse_query(query)
        
        # Enhance for retrieval
        enhanced_phrases = self.enhance_for_retrieval(structured_data)
        structured_data["enhanced_search_phrases"] = enhanced_phrases
        
        return structured_data

# Initialize parser instance
query_parser = QueryParser()

# Utility functions for integration
def parse_insurance_query(query: str) -> Dict:
    """Main function to parse insurance queries"""
    return query_parser.process_query(query)

def get_search_terms(query: str) -> list:
    """Extract search terms for vector retrieval"""
    parsed = parse_insurance_query(query)
    return parsed.get("enhanced_search_phrases", [query])

# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "I'm a 46-year-old male needing knee surgery in Pune. Does my 3-month policy cover this?",
        "What are the exclusions for dental procedures?",
        "Can I claim maternity benefits after 2 years of policy?",
        "Premium calculation for 35-year-old female in Mumbai"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = parse_insurance_query(query)
        print(f"Parsed: {json.dumps(result, indent=2)}")
        print(f"Search terms: {get_search_terms(query)}")
