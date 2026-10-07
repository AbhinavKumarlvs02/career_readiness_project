from langchain_core.prompts import ChatPromptTemplate

RESUME_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical recruiter and AI parser. Extract the candidate's skills, experience, and projects from the provided resume text. "
               "Standardize all technology names (e.g., 'ReactJS' and 'React.js' should both be 'React'). "
               "Do not invent or assume skills that are not explicitly stated or clearly implied by the project descriptions."),
    ("human", "{resume_text}")
])

JD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR analyst. Extract the core requirements from the following job description. "
               "Distinguish clearly between mandatory and preferred skills. Extract soft skills into a separate list. "
               "Identify the expected experience level and the exact role title."),
    ("human", "{jd_text}")
])