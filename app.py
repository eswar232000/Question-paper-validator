import streamlit as st

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

            # DISPLAY
            st.success("Validation Completed Successfully")

            st.dataframe(result_df, use_container_width=True)

            # STATISTICS
            st.subheader("Validation Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Questions", len(result_df))
            col2.metric(
                "Average Confidence",
                f"{round(result_df['Confidence (%)'].mean(), 2)}%"
            )
            col3.metric(
                "Good Quality Questions",
                len(result_df[result_df['Quality'] == 'Good'])
            )

            # DOWNLOAD
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Validation Report",
                data=csv,
                file_name="validation_report.csv",
                mime="text/csv"
            )

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("About Application")

st.sidebar.info(
    """
    Lightweight AI-based Question Paper Validator

    Features:
    - Upload CSV files
    - Bloom's Taxonomy Detection
    - CO Alignment Validation
    - CPU-only Inference
    - Streamlit Deployment
    - Hugging Face Lightweight Model
    """
)