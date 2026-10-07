from typing import List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from src.extraction.agent_prompts import RESUME_EXTRACTION_PROMPT

class Project(BaseModel):
    title: str = Field(description="Title of the project")
    tech_stack: List[str] = Field(description="List of technologies, frameworks, and languages used")
    description: str = Field(description="Brief summary of what was built and its impact")

class Experience(BaseModel):
    role: str = Field(description="Job title or role")
    company: str = Field(description="Company or organization name")
    duration: str = Field(description="Time period worked")
    tech_stack: List[str] = Field(description="Technologies used in this role")

class ResumeData(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    hard_skills: List[str] = Field(description="Explicit programming languages, tools, and frameworks mentioned")
    soft_skills: List[str] = Field(description="Soft skills inferred from bullet points (e.g., Leadership, Agile, Communication)")
    projects: List[Project] = Field(description="List of technical projects")
    experience: List[Experience] = Field(description="List of work experiences or internships")

def extract_resume_data(pdf_path: str, model_name: str = "openai/gpt-oss-120b") -> ResumeData:
    """Parses a PDF resume and extracts structured Pydantic data using the Groq API."""
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    resume_text = "\n".join([page.page_content for page in pages])
    
    # 2. Initialize Groq LLM
    # Groq runs Llama 3 natively. We use temperature 0 for strict factual extraction.
    llm = ChatGroq(model=model_name, temperature=0)
    
    # Bind Pydantic schema
    structured_llm = llm.with_structured_output(ResumeData)
    
    # 3. Create execution chain and invoke
    extraction_chain = RESUME_EXTRACTION_PROMPT | structured_llm
    parsed_data = extraction_chain.invoke({"resume_text": resume_text})
    
    return parsed_data