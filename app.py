import streamlit as st
import pandas as pd
import fitz
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIGimport streamlit as st
import pandas as pd
import fitz
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Question Paper Quality Validator",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------
# BLOOM TAXONOMY
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

            if re.search(rf"\b{verb}\b", q):

                return level

    return "Unknown"

# ---------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------
def extract_text_from_pdf(uploaded_file):

    text = ""

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf_document:

        text += page.get_text() + "\n"

    return text

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

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    best_index = similarity_scores.argmax()

    best_co = course_outcomes[
        best_index
    ]

    confidence = round(
        similarity_scores[
            best_index
        ] * 100,
        2
    )

    return best_co, confidence

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title(
    "📘 Question Paper Quality Validator"
)

co_file = st.file_uploader(
    "Upload Course Outcomes PDF",
    type=["pdf"]
)

question_file = st.file_uploader(
    "Upload Question Bank PDF",
    type=["pdf"]
)

# ---------------------------------------------------
# VALIDATION
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

        co_text = extract_text_from_pdf(
            co_file
        )

        q_text = extract_text_from_pdf(
            question_file
        )

        course_outcomes = [

            line.strip()

            for line in co_text.split("\n")

            if len(line.strip()) > 5
        ]

        questions = [

            line.strip()

            for line in q_text.split("\n")

            if (
                "?" in line
                or len(line.strip()) > 20
            )
        ]

        results = []

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

            if confidence < 40:

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

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Report",
            data=csv,
            file_name="validation_report.csv",
            mime="text/csv"
        )
# ---------------------------------------------------
st.set_page_config(
    page_title="Question Paper Quality Validator",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------
# BLOOM TAXONOMY
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

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf_document:

        text += page.get_text() + "\n"

    return text

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

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    best_index = similarity_scores.argmax()

    best_co = course_outcomes[
        best_index
    ]

    confidence = round(
        similarity_scores[
            best_index
        ] * 100,
        2
    )

    return best_co, confidence

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title(
    "📘 Question Paper Quality Validator"
)

st.markdown("""

### Features

- Bloom's Taxonomy Detection
- CO Alignment Validation
- Lightweight AI Similarity Analysis
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
# VALIDATION
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

        with st.spinner(
            "Extracting PDF text..."
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
        # RESULTS
        # -------------------------------------------
        results = []

        with st.spinner(
            "Running Validation..."
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

                if confidence < 40:

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
