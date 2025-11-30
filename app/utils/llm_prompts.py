import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Correct usage: DO NOT pass api_key here
client = OpenAI()  

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SCORECARD_SYSTEM = "You are an expert recruiter assistant. Be concise and factual."


def generate_scorecard(job_description: str, resume_text: str, candidate_name: str):
    """
    Generates an evaluation using OpenAI Chat API.
    If API fails → ALWAYS return: 'Your resume ranking is done.'
    """

    # Safety: blank text fallback
    if not resume_text or resume_text.strip() == "":
        return "Your resume ranking is done."

    try:
        prompt = f"""
You are an HR assistant. Evaluate the resume strictly against the job description.

Provide EXACTLY:

1) A numeric fit score (0-100) + one-sentence reason.
2) Top 5 skills from the resume that match the JD.
3) Two short interview questions tailored to the candidate.

Job Description:
{job_description}

Resume:
{resume_text}
"""

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

    except Exception:
        return "Your resume ranking is done."
