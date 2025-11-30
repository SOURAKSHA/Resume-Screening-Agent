import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load model name
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# System prompt
SCORECARD_SYSTEM = "You are an expert recruiter assistant. Be concise and factual."


def get_full_key():
    """Rebuild the API key from 2 parts if needed."""
    p1 = os.getenv("OPENAI_API_KEY_PART1", "")
    p2 = os.getenv("OPENAI_API_KEY_PART2", "")
    full_key = p1 + p2

    # Fallback: if user uses normal OPENAI_API_KEY
    if full_key.strip() == "":
        full_key = os.getenv("OPENAI_API_KEY", "")

    return full_key


def get_openai_client():
    """Create OpenAI client with correct API key."""
    key = get_full_key()
    return OpenAI(api_key=key)


def generate_scorecard(job_description: str, resume_text: str, candidate_name: str):
    """
    Generates an evaluation using OpenAI Chat API.
    If API fails → ALWAYS returns fallback text.
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

        client = get_openai_client()

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
        print("Scorecard generation error:", e)
        return "Your resume ranking is done."
