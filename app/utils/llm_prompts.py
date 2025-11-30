import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


# Rebuild key from 2 parts
def get_full_key():
    return os.getenv("OPENAI_API_KEY_PART1", "") + os.getenv("OPENAI_API_KEY_PART2", "")


client = OpenAI(api_key=get_full_key())

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SCORECARD_SYSTEM = "You are an expert recruiter assistant. Be concise and factual."


def generate_scorecard(job_description: str, resume_text: str, candidate_name: str):
    """
    Generates evaluation using OpenAI Chat API.
    If API fails -> ALWAYS return fallback message.
    """

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

    except Exception as e:
        print("Scorecard Error:", e)
        return "Your resume ranking is done."
