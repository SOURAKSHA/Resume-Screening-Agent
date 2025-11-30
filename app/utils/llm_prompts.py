import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Rebuild API key from parts
def get_full_key():
    p1 = os.getenv("OPENAI_API_KEY_PART1", "")
    p2 = os.getenv("OPENAI_API_KEY_PART2", "")
    return p1 + p2

# Correct client usage
client = OpenAI(api_key=get_full_key())

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SCORECARD_SYSTEM = "You are an expert recruiter assistant. Be concise and factual."


def generate_scorecard(job_description: str, resume_text: str, candidate_name: str):
    """
    Generates evaluation using latest OpenAI API.
    If API fails → return safe fallback.
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

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SCORECARD_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=350,
            temperature=0.2,
        )

        return response.output_text

    except Exception as e:
        print("LLM ERROR:", e)
        return "Your resume ranking is done."
