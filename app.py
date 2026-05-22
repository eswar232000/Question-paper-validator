import streamlit as st
import pandas as pd
import fitz
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Question Paper Quality Validator",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=-1
    )

    return classifier

classifier = load_model()

# -----------------------------
# BLOOM LEVELS
# -----------------------------
blooms_keywords = {
    "Remember": ["define", "list", "state", "identify", "recall"],
    "Understand": ["explain", "describe", "summarize", "discuss"],
    "Apply": ["solve", "implement", "use", "demonstrate"],
    "Analyze": ["analyze", "compare", "differentiate", "classify"],
    "Evaluate": ["evaluate", "justify", "critique", "assess"],
    "Create": ["design", "develop", "construct", "propose"]
}

# -----------------------------
# DETECT BLOOM LEVEL
# -----------------------------
def detect_blooms_level(question):

    q = question.lower()

    for level, verbs in blooms_keywords.items():

        for verb in verbs:

            if verb in q:
                return level

    return "Unknown"

# -----------------------------
# VALIDATE CO ALIGNMENT
# -----------------------------
def validate_co_alignment(question, course_outcomes):

    result = classifier(
        question,
        candidate_labels=course_outcomes,
        multi_label=False
    )

    best_co = result["labels"][0]
    confidence = round(result["scores"][0] * 100, 2)

    return best_co, confidence

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(uploaded_file):

    text = ""

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf_document:
        text += page.get_text()

    return text

# -----------------------------
# UI
# -----------------------------
st.title("📘 Question Paper Quality Validator")

co_file = st.file_uploader(
    "Upload Course Outcomes PDF",
    type=["pdf"]
)

question_file = st.file_uploader(
    "Upload Question Bank PDF",
    type=["pdf"]
)

# -----------------------------
# VALIDATION
# -----------------------------
if st.button("Validate Question Paper"):

    if co_file is None or question_file is None:

        st.warning("Please upload both PDF files")

    else:

        co_text = extract_text_from_pdf(co_file)
        q_text = extract_text_from_pdf(question_file)

        course_outcomes = [
            line.strip()
            for line in co_text.split("\n")
            if line.strip()
        ]

        questions = [
            line.strip()
            for line in q_text.split("\n")
            if line.strip()
        ]

        results = []

        with st.spinner("Running AI Validation..."):

            for question in questions:

                blooms_level = detect_blooms_level(question)

                best_co, confidence = validate_co_alignment(
                    question,
                    course_outcomes
                )

                quality = "Good"

                if confidence < 50:
                    quality = "Needs Improvement"

                results.append({
                    "Question": question,
                    "Bloom Level": blooms_level,
                    "Aligned CO": best_co,
                    "Confidence (%)": confidence,
                    "Quality": quality
                })

        result_df = pd.DataFrame(results)

        st.success("Validation Completed")

        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Report",
            data=csv,
            file_name="validation_report.csv",
            mime="text/csv"
        )
