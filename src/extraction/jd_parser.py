from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.extraction.agent_prompts import JD_EXTRACTION_PROMPT

class JobDescriptionData(BaseModel):
    role_title: str = Field(description="The job title being hired for")
    mandatory_skills: List[str] = Field(description="Hard skills explicitly required by the job")
    preferred_skills: List[str] = Field(description="Skills that are nice-to-have or preferred")
    soft_skills_required: List[str] = Field(description="Required soft skills (e.g., Problem Solving, Teamwork)")
    experience_level: str = Field(description="Expected experience level (e.g., Entry-Level, 3-5 years)")

def extract_jd_data(jd_text: str, model_name: str = "openai/gpt-oss-120b") -> JobDescriptionData:
    """Parses unstructured Job Description text into structured Pydantic data using the Groq API."""
    # 1. Initialize Groq LLM
    llm = ChatGroq(model=model_name, temperature=0)
    
    # Bind Pydantic schema
    structured_llm = llm.with_structured_output(JobDescriptionData)
    
    # 2. Create execution chain and invoke
    extraction_chain = JD_EXTRACTION_PROMPT | structured_llm
    parsed_data = extraction_chain.invoke({"jd_text": jd_text})
    
    return parsed_data