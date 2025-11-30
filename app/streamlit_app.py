

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from utils.parser import parse_resume
from utils.ranking import index_resume, rank_resumes, RESUME_STORE
from utils.llm_prompts import generate_scorecard
import pandas as pd

st.set_page_config(page_title="ResumeRanker Pro", layout="wide")

st.title("ResumeRanker Pro — Resume Screening Agent")
st.markdown("Upload resumes (PDF/DOCX). Paste job description. Click **Rank Resumes**.")

uploaded_files = st.file_uploader("Upload resumes (multiple)", accept_multiple_files=True, type=['pdf','docx','doc','txt'])
job_description = st.text_area("Paste Job Description here", height=200)


if st.button("Index uploaded resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        for f in uploaded_files:
            parsed = parse_resume(f)
            index_resume(
                parsed["text"],
                parsed["filename"],
                metadata={"email": parsed["email"], "name": parsed["name"]}
            )
        st.success("All resumes indexed successfully!")


if st.button("Rank Resumes"):
    if not job_description:
        st.warning("Please add a job description.")
    else:
        if not RESUME_STORE:
            st.error("No resumes indexed yet!")
        else:
            st.info("Ranking resumes... ⏳")
            results = rank_resumes(job_description)

            st.subheader("Ranked Resumes")

            cards = []
            for r in results:
                resume_txt = "Resume text stored internally"
                try:
                    sc = generate_scorecard(job_description, resume_txt, r['filename'])
                except:
                    sc = "LLM scorecard unavailable"

                cards.append({
                    "filename": r["filename"],
                    "score": r["score"],
                    "scorecard": sc
                })

            df = pd.DataFrame(cards)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="resume_rankings.csv",
                mime="text/csv"
            )       