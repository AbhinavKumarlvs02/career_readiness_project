import os
import json
from dotenv import load_dotenv

# 1. LOAD THIS FIRST before importing anything else
load_dotenv()

# 2. NOW import the parsers
from src.extraction.resume_parser import extract_resume_data
from src.extraction.jd_parser import extract_jd_data

def run_test():
    # Define paths and inputs
    sample_pdf_path = "sample_resume.pdf" # Ensure this file exists in your folder
    
    sample_jd_text = """
    We are looking for a Full Stack Developer to join our team. 
    You must have at least 1-2 years of experience building web applications.
    Mandatory Skills: JavaScript, React.js, Node.js, and PostgreSQL.
    Preferred Skills: Docker, AWS, and Tailwind CSS.
    The ideal candidate should have strong problem-solving skills, be a team player, 
    and have experience working in Agile/Scrum environments.
    """

    print("--- Starting Extraction Test ---\n")

    # Test Resume Extraction
    try:
        print(f"Loading and extracting data from: {sample_pdf_path}...")
        resume_data = extract_resume_data(sample_pdf_path)
        
        print("\n✅ RESUME EXTRACTION SUCCESSFUL:")
        print(json.dumps(json.loads(resume_data.model_dump_json()), indent=2))
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{sample_pdf_path}'. Please put a test PDF in the folder.")
    except Exception as e:
        print(f"❌ Error extracting resume: {e}")

    print("\n" + "="*50 + "\n")

    # Test Job Description Extraction
    try:
        print("Extracting data from Job Description text...")
        jd_data = extract_jd_data(sample_jd_text)
        
        print("\n✅ JOB DESCRIPTION EXTRACTION SUCCESSFUL:")
        print(json.dumps(json.loads(jd_data.model_dump_json()), indent=2))
        
    except Exception as e:
        print(f"❌ Error extracting JD: {e}")

if __name__ == "__main__":
    # Ensure the GROQ API key is set before running
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ Warning: GROQ_API_KEY is not set in your environment variables or .env file.")
        print("Please add GROQ_API_KEY=\"gsk_...\" to your .env file before running this script.")
    else:
        run_test()