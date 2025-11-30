# ResumeRanker Pro — Resume Screening Agent

**Short:** ResumeRanker Pro takes a job description and resumes (PDF/DOCX), extracts text, creates embeddings, ranks candidates by similarity, and generates optional AI scorecards.

## Features
- Upload multiple resumes  
- Parse PDF & DOCX  
- Create embeddings (SentenceTransformer – MiniLM)  
- Rank resumes by similarity to a Job Description  
- Optional AI scorecard (OpenAI)  
- Clean fallback if no API credits (“Your resume ranking is done.”)  
- Export ranked results as CSV  

## Tech Stack
- Python, Streamlit (UI)  
- SentenceTransformer (embeddings)  
- NumPy (similarity scoring)  
- pdfplumber, python-docx (resume parsing)  
- OpenAI (optional scorecard generation)  

## Setup (Local)

1. **Clone repository**

```bash
git clone <your-repo-url>
cd resume-ranker-pro
Create & activate a virtual environment

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
Install dependencies

pip install -r requirements.txt
Create a .env file

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini


Run the Streamlit app

streamlit run app/streamlit_app.py