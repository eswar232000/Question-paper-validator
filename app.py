import streamlit as st
import pandas as pd
import fitz
import re

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Question Paper Quality Validator",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():

    model_name = "facebook/bart-large-mnli"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    classifier = pipeline(
        task="zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    return classifier


classifier = load_model()

# ---------------------------------------------------
# BLOOM TAXONOMY KEYWORDS
# ---------------------------------------------------
blooms_keywords = {

    "Remember": [
        "define",
        "list",
        "state",
        "identify",
        "recall"
    ],

    "Understand": [
        "explain",
        "describe",
        "summarize",
        "discuss"
    ],

    "Apply": [
        "solve",
        "implement",
        "use",
        "demonstrate"
    ],

    "Analyze": [
        "analyze",
        "compare",
        "differentiate",
        "classify"
    ],

    "Evaluate": [
        "evaluate",
        "justify",
        "critique",
        "assess"
    ],

    "Create": [
        "design",
        "develop",
        "construct",
        "propose"
    ]
}

# ---------------------------------------------------
# DETECT BLOOM LEVEL
# ---------------------------------------------------
def detect_blooms_level(question):

    q = question.lower()

    for level, verbs in blooms_keywords.items():

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

            page_text = page.get_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        st.error(
            f"PDF Extraction Error: {e}"
        )

    return text

# ---------------------------------------------------
# VALIDATE CO ALIGNMENT
# ---------------------------------------------------
def validate_co_alignment(
    question,
    course_outcomes
):

    try:

        result = classifier(
            sequences=question,
            candidate_labels=course_outcomes,
            multi_label=False
        )

        best_co = result["labels"][0]

        confidence = round(
            result["scores"][0] * 100,
            2
        )

        return best_co, confidence

    except Exception as e:

        return (
            f"Model Error: {e}",
            0
        )

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title(
    "📘 Question Paper Quality Validator"
)

st.markdown("""

### Features

- Bloom's Taxonomy Detection
- Course Outcome Alignment
- AI-based Validation
- CSV Report Generation

""")

# ---------------------------------------------------
# FILE UPLOADS
# ---------------------------------------------------
co_file = st.file_uploader(
    "Upload Course Outcomes PDF",
    type=["pdf"]
)

question_file = st.file_uploader(
    "Upload Question Bank PDF",
    type=["pdf"]
)

# ---------------------------------------------------
# VALIDATE
# ---------------------------------------------------
if st.button(
    "Validate Question Paper"
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
        # EXTRACT TEXT
        # -------------------------------------------
        with st.spinner(
            "Extracting PDF content..."
        ):

            co_text = extract_text_from_pdf(
                co_file
            )

            q_text = extract_text_from_pdf(
                question_file
            )

        # -------------------------------------------
        # COURSE OUTCOMES
        # -------------------------------------------
        course_outcomes = [

            line.strip()

            for line in co_text.split("\n")

            if len(
                line.strip()
            ) > 5
        ]

        # -------------------------------------------
        # QUESTIONS
        # -------------------------------------------
        questions = [

            line.strip()

            for line in q_text.split("\n")

            if (
                "?" in line
                or len(
                    line.strip()
                ) > 20
            )
        ]

        # -------------------------------------------
        # CHECKS
        # -------------------------------------------
        if len(
            course_outcomes
        ) == 0:

            st.error(
                "No Course Outcomes detected"
            )

            st.stop()

        if len(
            questions
        ) == 0:

            st.error(
                "No Questions detected"
            )

            st.stop()

        # -------------------------------------------
        # PROCESS
        # -------------------------------------------
        results = []

        with st.spinner(
            "Running AI Validation..."
        ):

            for question in questions:

                blooms_level = (
                    detect_blooms_level(
                        question
                    )
                )

                best_co, confidence = (
                    validate_co_alignment(
                        question,
                        course_outcomes
                    )
                )

                quality = "Good"

                if confidence < 50:

                    quality = (
                        "Needs Improvement"
                    )

                results.append({

                    "Question":
                        question,

                    "Bloom Level":
                        blooms_level,

                    "Aligned CO":
                        best_co,

                    "Confidence (%)":
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
        # DOWNLOAD CSV
        # -------------------------------------------
        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Report",
            data=csv,
            file_name=(
                "validation_report.csv"
            ),
            mime="text/csv"
        )
