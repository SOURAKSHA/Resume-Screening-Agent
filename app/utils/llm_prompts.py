import os
from openai import OpenAI


# Build OpenAI client from Streamlit Secrets
def get_llm_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    return OpenAI(api_key=key)


client = get_llm_client()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SCORECARD_SYSTEM = "You are an expert recruiter assistant. Be concise and factual."


def generate_scorecard(job_description: str, resume_text: str, candidate_name: str):
    """
    Generates scorecard using OpenAI Chat API.
    """
    if not resume_text or resume_text.strip() == "":
        return "Your resume ranking is done."

    prompt = f"""
Evaluate the resume strictly against the job description.

Return EXACTLY:

1) A numeric fit score (0–100) + one-sentence reason
2) Top 5 matching skills from the resume
3) 2 interview questions

Job Description:
{job_description}

Resume:
{resume_text}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SCORECARD_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=350,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print("Scorecard Error:", e)
        return "Your resume ranking is done."
