import streamlit as st
import pandas as pd
import fitz
import re
import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Question Paper Validator",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📘 AI Question Paper Validator")

st.markdown("""

### Features

✅ Bloom Taxonomy Detection  
✅ CO Alignment Validation  
✅ Question Quality Analysis  
✅ Bloom Coverage Analysis  
✅ CSV Report Generation  

""")

# ---------------------------------------------------
# BLOOM VERBS
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
# PDF EXTRACTION
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
            f"PDF Error: {e}"
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

            course_outcomes.append(cleaned)

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
# CO ALIGNMENT
# ---------------------------------------------------
def validate_co_alignment(
    question,
    course_outcomes
):

    documents = [question] + course_outcomes

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarities = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
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
# QUALITY ANALYSIS
# ---------------------------------------------------
def evaluate_quality(
    bloom_level,
    confidence
):

    score = 0

    if bloom_level != "Unknown":

        score += 50

    if confidence >= 80:

        score += 50

    elif confidence >= 60:

        score += 35

    elif confidence >= 40:

        score += 20

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

        with st.spinner(
            "Reading PDFs..."
        ):

            co_text = extract_text_from_pdf(
                co_file
            )

            q_text = extract_text_from_pdf(
                question_file
            )

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

        if len(course_outcomes) == 0:

            st.error(
                "No Course Outcomes found"
            )

            st.stop()

        if len(questions) == 0:

            st.error(
                "No Questions found"
            )

            st.stop()

        results = []

        with st.spinner(
            "Running Validation..."
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

                    "Similarity Score (%)":
                        confidence,

                    "Quality":
                        quality
                })

        result_df = pd.DataFrame(
            results
        )

        st.success(
            "Validation Completed"
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
            "Coverage",
            f"{coverage}%"
        )

        if len(missing) > 0:

            st.warning(
                f"Missing Levels: {', '.join(missing)}"
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
