import streamlit as st
import pandas as pd
import fitz
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="GenAI Question Paper Validator",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        device="cpu"
    )

    return model


model = load_model()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📘 GenAI Question Paper Validator")

st.markdown("""

### Features

✅ GenAI Semantic CO Alignment  
✅ Bloom Taxonomy Validation  
✅ Bloom Coverage Analysis  
✅ AI Similarity Scoring  
✅ Question Quality Evaluation  
✅ Lightweight CPU-Based AI Model  

""")

# ---------------------------------------------------
# BLOOM TAXONOMY
# ---------------------------------------------------
blooms_taxonomy = {

    "Remember": [
        "define",
        "list",
        "state",
        "identify",
        "recall",
        "name"
    ],

    "Understand": [
        "explain",
        "describe",
        "summarize",
        "discuss",
        "illustrate"
    ],

    "Apply": [
        "apply",
        "solve",
        "implement",
        "use",
        "demonstrate",
        "show"
    ],

    "Analyze": [
        "analyze",
        "compare",
        "differentiate",
        "classify",
        "examine"
    ],

    "Evaluate": [
        "evaluate",
        "justify",
        "critique",
        "assess",
        "validate"
    ],

    "Create": [
        "design",
        "develop",
        "construct",
        "propose",
        "create"
    ]
}

# ---------------------------------------------------
# DETECT BLOOM LEVEL
# ---------------------------------------------------
def detect_bloom_level(question):

    q = question.lower()

    for level, verbs in blooms_taxonomy.items():

        for verb in verbs:

            if re.search(
                rf"\b{verb}\b",
                q
            ):

                return level

    return "Unknown"

# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------
def extract_text_from_pdf(uploaded_file):

    text = ""

    try:

        pdf_document = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        for page in pdf_document:

            text += page.get_text() + "\n"

    except Exception as e:

        st.error(
            f"PDF Extraction Error: {e}"
        )

    return text

# ---------------------------------------------------
# EXTRACT COURSE OUTCOMES
# ---------------------------------------------------
def extract_course_outcomes(text):

    lines = text.split("\n")

    course_outcomes = []

    for line in lines:

        cleaned = line.strip()

        if cleaned.startswith("CO"):

            course_outcomes.append(
                cleaned
            )

    return course_outcomes

# ---------------------------------------------------
# EXTRACT QUESTIONS
# ---------------------------------------------------
def extract_questions(text):

    lines = text.split("\n")

    questions = []

    for line in lines:

        cleaned = line.strip()

        if (
            len(cleaned) > 15
            and (
                "?" in cleaned
                or any(
                    verb in cleaned.lower()

                    for verbs in blooms_taxonomy.values()

                    for verb in verbs
                )
            )
        ):

            questions.append(cleaned)

    return questions

# ---------------------------------------------------
# SEMANTIC CO ALIGNMENT
# ---------------------------------------------------
def validate_co_alignment(
    question,
    course_outcomes
):

    question_embedding = model.encode(
        [question]
    )

    co_embeddings = model.encode(
        course_outcomes
    )

    similarities = cosine_similarity(
        question_embedding,
        co_embeddings
    )[0]

    best_index = np.argmax(
        similarities
    )

    best_co = course_outcomes[
        best_index
    ]

    confidence = round(
        similarities[
            best_index
        ] * 100,
        2
    )

    return best_co, confidence

# ---------------------------------------------------
# QUALITY EVALUATION
# ---------------------------------------------------
def evaluate_quality(
    bloom_level,
    confidence
):

    score = 0

    if bloom_level != "Unknown":

        score += 40

    if confidence >= 80:

        score += 60

    elif confidence >= 60:

        score += 45

    elif confidence >= 40:

        score += 30

    if score >= 90:

        return "Excellent"

    elif score >= 70:

        return "Good"

    elif score >= 50:

        return "Moderate"

    return "Needs Improvement"

# ---------------------------------------------------
# BLOOM COVERAGE
# ---------------------------------------------------
def bloom_coverage(results):

    present = set()

    for row in results:

        present.add(
            row["Bloom Level"]
        )

    total_levels = 6

    coverage = round(
        (
            len(present)
            / total_levels
        ) * 100,
        2
    )

    missing = [

        level

        for level in blooms_taxonomy.keys()

        if level not in present
    ]

    return coverage, missing

# ---------------------------------------------------
# FILE UPLOADERS
# ---------------------------------------------------
co_file = st.file_uploader(
    "Upload Course Outcomes PDF",
    type=["pdf"],
    key="co_pdf"
)

question_file = st.file_uploader(
    "Upload Question Paper PDF",
    type=["pdf"],
    key="question_pdf"
)

# ---------------------------------------------------
# VALIDATION BUTTON
# ---------------------------------------------------
if st.button(
    "Validate Question Paper",
    key="validate_button"
):

    if (
        co_file is None
        or question_file is None
    ):

        st.warning(
            "Please upload both PDF files"
        )

    else:

        # -------------------------------------------
        # EXTRACT PDF TEXT
        # -------------------------------------------
        with st.spinner(
            "Reading PDFs..."
        ):

            co_text = extract_text_from_pdf(
                co_file
            )

            q_text = extract_text_from_pdf(
                question_file
            )

        # -------------------------------------------
        # EXTRACT CONTENT
        # -------------------------------------------
        course_outcomes = (
            extract_course_outcomes(
                co_text
            )
        )

        questions = (
            extract_questions(
                q_text
            )
        )

        # -------------------------------------------
        # VALIDATION CHECKS
        # -------------------------------------------
        if len(course_outcomes) == 0:

            st.error(
                "No Course Outcomes Found"
            )

            st.stop()

        if len(questions) == 0:

            st.error(
                "No Questions Found"
            )

            st.stop()

        # -------------------------------------------
        # VALIDATION
        # -------------------------------------------
        results = []

        with st.spinner(
            "Running GenAI Validation..."
        ):

            for question in questions:

                bloom_level = (
                    detect_bloom_level(
                        question
                    )
                )

                best_co, confidence = (
                    validate_co_alignment(
                        question,
                        course_outcomes
                    )
                )

                quality = (
                    evaluate_quality(
                        bloom_level,
                        confidence
                    )
                )

                results.append({

                    "Question":
                        question,

                    "Bloom Level":
                        bloom_level,

                    "Aligned CO":
                        best_co,

                    "Semantic Similarity (%)":
                        confidence,

                    "Quality":
                        quality
                })

        # -------------------------------------------
        # DATAFRAME
        # -------------------------------------------
        result_df = pd.DataFrame(
            results
        )

        st.success(
            "Validation Completed Successfully"
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # -------------------------------------------
        # BLOOM COVERAGE
        # -------------------------------------------
        coverage, missing = (
            bloom_coverage(
                results
            )
        )

        st.subheader(
            "📊 Bloom Taxonomy Coverage"
        )

        st.metric(
            "Coverage Percentage",
            f"{coverage}%"
        )

        if len(missing) > 0:

            st.warning(
                f"Missing Bloom Levels: {', '.join(missing)}"
            )

        else:

            st.success(
                "All Bloom Levels Covered"
            )

        # -------------------------------------------
        # DOWNLOAD CSV
        # -------------------------------------------
        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Validation Report",
            data=csv,
            file_name="validation_report.csv",
            mime="text/csv",
            key="download_csv"
        )
