import io
import re
import pdfplumber
from docx import Document

def extract_text_from_pdf(path_or_fileobj):
    text_parts = []
    with pdfplumber.open(path_or_fileobj) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)

def extract_text_from_docx(path_or_fileobj):
    if hasattr(path_or_fileobj, "read"):
        doc = Document(io.BytesIO(path_or_fileobj.read()))
    else:
        doc = Document(path_or_fileobj)
    texts = []
    for p in doc.paragraphs:
        if p.text:
            texts.append(p.text)
    return "\n".join(texts)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def parse_resume(uploaded_file):
    filename = getattr(uploaded_file, "name", None) or str(uploaded_file)
    text = ""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif filename.lower().endswith(".docx") or filename.lower().endswith(".doc"):
        text = extract_text_from_docx(uploaded_file)
    else:
        try:
            if hasattr(uploaded_file, "read"):
                uploaded_file.seek(0)
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            else:
                with open(uploaded_file, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception:
            text = ""
    text = clean_text(text)
    email = extract_email(text)
    name = infer_name_from_filename(filename)
    years_exp = extract_years_experience(text)
    return {
        "filename": filename,
        "text": text,
        "email": email,
        "name": name,
        "years_experience": years_exp
    }

def extract_email(text):
    m = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return m.group(0) if m else None

def infer_name_from_filename(filename):
    import os
    base = os.path.splitext(os.path.basename(filename))[0]
    base = base.replace("_", " ").replace("-", " ").strip()
    return base.title() if base else "Unknown"

def extract_years_experience(text):
    m = re.search(r"(\d{1,2})\+?\s+(?:years|yrs)\b", text.lower())
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None
